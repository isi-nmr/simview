"""Resolve loop spans in a precompiled Bruker pulse program."""

import re
from math import gcd

import numpy as np


def resolve_loop_count(token: str, structure: dict, aliases: dict[str, str]) -> int | str:
    for name in (token, aliases.get(token, token)):
        value = structure.get("loop_values", {}).get(name, structure.get("parameters", {}).get(name, name))
        if isinstance(value, int) or isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str) and re.fullmatch(r"\d+", value):
            return int(value)
    return aliases.get(token, token)


def ppg_frequency_offsets(
    records: list[dict], loops: dict[int, dict], lists: dict[str, np.ndarray],
) -> dict[int, list[tuple[str, str, str, float, float]]]:
    """Find programmed Tx/Rx list offsets at their use sites and in counted loops."""
    if not lists:
        return {}

    codes = [str(record["text"]).partition(";")[0] for record in records]
    operations = [
        [(name.casefold(), operation.lower()) for name, operation in re.findall(
            r"\b(\w+)\.(inc|res|store|restore)\b", code, re.IGNORECASE,
        ) if name.casefold() in lists]
        for code in codes
    ]
    strides = {}
    for end, loop in loops.items():
        start = loop.get("start_record_index")
        count = loop["count"]
        if start is None or not isinstance(count, int) or count < 2:
            continue
        advances = {}
        resets = set()
        for index in range(start, end):
            for name, operation in operations[index]:
                if operation == "inc":
                    advances[name] = advances.get(name, 0) + 1
                elif operation in {"res", "restore"}:
                    resets.add(name)
            for name, stride in strides.get(index, {}).items():
                advances[name] = advances.get(name, 0) + stride * (loops[index]["count"] - 1)
        strides[end] = {name: stride for name, stride in advances.items() if name not in resets}

    cursors = dict.fromkeys(lists, 0)
    saved = cursors.copy()
    offsets: dict[int, list[tuple[str, str, str, float, float]]] = {}
    for index, code in enumerate(codes):
        for symbol, receive, channel in re.findall(r"\b(\w+)\s*(\(\s*receive\s*\))?\s*:\s*f(\d+)\b", code, re.IGNORECASE):
            name = symbol.casefold()
            if name not in lists:
                continue
            values = lists[name]
            enclosing = [
                end for end, advances in strides.items()
                if name in advances and loops[end]["start_record_index"] <= index < end
            ]
            if enclosing:
                end = min(enclosing, key=lambda stop: stop - loops[stop]["start_record_index"])
                stride = strides[end][name]
                count = min(loops[end]["count"], len(values) // gcd(stride, len(values)))
                indices = (cursors[name] + stride * np.arange(count)) % len(values)
                selected = values[indices]
            else:
                selected = values[[cursors[name] % len(values)]]
            offsets.setdefault(index, []).append(
                ("Rx" if receive else "Tx", f"f{channel}", symbol, float(np.min(selected)), float(np.max(selected))),
            )
        for name, operation in operations[index]:
            if operation == "inc":
                cursors[name] += 1
            elif operation == "res":
                cursors[name] = 0
            elif operation == "store":
                saved[name] = cursors[name]
            else:
                cursors[name] = saved[name]
        for name, stride in strides.get(index, {}).items():
            cursors[name] += stride * (loops[index]["count"] - 1)
    return offsets


def collect_ppg_loops(records: list[dict], structure: dict, aliases: dict[str, str]) -> dict[int, dict]:
    codes = [str(record["text"]).partition(";")[0].strip() for record in records]
    labels: dict[str, list[int]] = {}
    for index, code in enumerate(codes):
        label = re.match(r"(\w+)\s*,", code)
        if label:
            labels.setdefault(label[1], []).append(index)

    loops = {}
    braces: list[tuple[int, int | None]] = []
    previous: int | None = None
    for index, code in enumerate(codes):
        if not code:
            continue
        if code == "{":
            braces.append((index, previous))
        elif code == "}" and braces:
            opening, opener_index = braces.pop()
            opener = codes[opener_index] if opener_index is not None else ""
            block_loop = re.fullmatch(r"loop\s+(\w+)", opener, re.IGNORECASE)
            if block_loop:
                token = block_loop[1]
                loops[index] = {
                    "target": token, "count_token": token, "count": resolve_loop_count(token, structure, aliases),
                    "start_record_index": opening + 1, "source_record_index": opener_index,
                }
        else:
            counted = re.search(r"\blo\s+to\s+(\w+)\s+times\s+(\w+)", code, re.IGNORECASE)
            jump = re.search(r'\bgoto\s+(\w+)\s*$', code, re.IGNORECASE)
            if counted or jump:
                target = (counted or jump)[1]
                start = next((i for i in reversed(labels.get(target, [])) if i < index), None)
                if counted:
                    token = counted[2]
                    count = resolve_loop_count(token, structure, aliases)
                elif start is not None:
                    conditions = [codes[opener] for _, opener in braces if opener is not None]
                    token, count = resolve_backward_jump_count(
                        codes, labels, start, index, conditions, structure, aliases,
                    )
                else:
                    previous = index
                    continue
                loops[index] = {
                    "target": target, "count_token": token, "count": count,
                    "start_record_index": start, "source_record_index": index,
                }
        previous = index
    return loops


def resolve_backward_jump_count(
    codes: list[str], labels: dict[str, list[int]], start: int, end: int,
    conditions: list[str], structure: dict, aliases: dict[str, str],
) -> tuple[str, int | str | None]:
    """Resolve the common countdown loop with a forward exit and backward jump."""
    direct_condition = re.match(r'if\s+"([^"]+)"\s+goto\b', codes[end], re.IGNORECASE)
    if direct_condition:
        countdown = re.fullmatch(r"(\w+)\s*>\s*=\s*0", direct_condition[1])
        if countdown:
            counter = countdown[1]
            update = next((code for code in reversed(codes[start:end]) if code), "")
            if re.fullmatch(rf'"{counter}\s*=\s*{counter}\s*-\s*1"', update):
                return counter, resolve_loop_count(counter, structure, aliases)
        return direct_condition[1], None
    for opener in reversed(conditions):
        condition = re.fullmatch(r'if\s+"([^"]+)"', opener, re.IGNORECASE)
        if condition:
            countdown = re.fullmatch(r"(\w+)\s*>=\s*0", condition[1])
            if countdown:
                counter = countdown[1]
                decrement = re.compile(rf'"{counter}\s*=\s*{counter}\s*-\s*1"')
                preceding = next((i for i in range(end - 1, start, -1) if codes[i] == opener), None)
                if preceding is not None:
                    update = next((code for code in reversed(codes[start:preceding]) if code), "")
                    if decrement.fullmatch(update):
                        return counter, resolve_loop_count(counter, structure, aliases)
            return condition[1], None

    for index in range(start + 1, end):
        exit_jump = re.fullmatch(r'if\s+"(\w+)\s*<\s*(\d+)"\s+goto\s+(\w+)', codes[index], re.IGNORECASE)
        if not exit_jump or not any(position > end for position in labels.get(exit_jump[3], [])):
            continue
        counter, limit = exit_jump[1], int(exit_jump[2])
        update = re.compile(rf'"{counter}\s*=\s*{counter}\s*-\s*1"')
        if sum(bool(update.fullmatch(code)) for code in codes[index + 1:end]) != 1:
            continue
        initializer = re.compile(rf'"{counter}\s*=\s*(\w+)"')
        for code in reversed(codes[:start]):
            match = initializer.fullmatch(code)
            if match:
                initial_count = resolve_loop_count(match[1], structure, aliases)
                if isinstance(initial_count, int):
                    return match[1], max(1, initial_count - limit + 2)
                return match[1], initial_count
    return "runtime", None
