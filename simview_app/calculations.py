import re

import numpy as np


class CalculationMixin:
    def sanitize_filename(self, value: str) -> str:
        sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._")
        return sanitized or "plot"

    def get_channel_axis_label(self, channel: list[dict]) -> tuple[str, str | None]:
        label = str(channel[0].get("chanLabel", ""))
        units = str(channel[0].get("units", "")).strip() or None
        if "(" in label and ")" in label:
            return label, None
        return label, units

    def is_percent_gradient_units(self, units: str | None) -> bool:
        normalized = (units or "").strip().lower()
        return normalized in {"", "%", "percent", "pct"}

    def hz_per_mm_to_mt_per_m(self, data: np.ndarray) -> np.ndarray:
        if self.nucleusGammaMHzPerT <= 0:
            return np.zeros_like(np.asarray(data, dtype=float))
        return np.asarray(data, dtype=float) / self.nucleusGammaMHzPerT

    def hz_per_mm_to_t_per_m(self, data: np.ndarray) -> np.ndarray:
        return self.hz_per_mm_to_mt_per_m(data) * 1e-3

    def get_gradient_display_units(self, raw_units: str | None) -> str:
        if self.gradientCalibrationHzPerMm > 0 and self.is_percent_gradient_units(raw_units):
            if self.displayGradientsInMtPerM:
                return "mT/m"
            return "Hz/mm"
        return (raw_units or "").strip() or "%"

    def scale_gradient_data(self, data: np.ndarray, raw_units: str | None) -> np.ndarray:
        scaled = np.asarray(data, dtype=float)
        if self.gradientCalibrationHzPerMm > 0 and self.is_percent_gradient_units(raw_units):
            scaled_hz_per_mm = scaled * (self.gradientCalibrationHzPerMm / 100.0)
            if self.displayGradientsInMtPerM:
                return self.hz_per_mm_to_mt_per_m(scaled_hz_per_mm)
            return scaled_hz_per_mm
        return scaled

    def get_gradient_physical_hz_per_mm(self, line: dict) -> np.ndarray | None:
        if self.gradientCalibrationHzPerMm <= 0:
            return None

        raw_units = str(line.get("raw_units", line.get("units", "%")))
        if not self.is_percent_gradient_units(raw_units):
            return None

        raw_data = np.asarray(line.get("raw_data", line.get("data", [])), dtype=float)
        return raw_data * (self.gradientCalibrationHzPerMm / 100.0)

    def update_gradient_channels(self) -> None:
        for channel in self.channels:
            for line in channel:
                if line.get("type") != "grads":
                    continue
                raw_data = np.asarray(line.get("raw_data", line.get("data", [])), dtype=float)
                raw_units = str(line.get("raw_units", line.get("units", "%")))
                line["raw_data"] = raw_data
                line["raw_units"] = raw_units
                line["physical_hz_per_mm"] = self.get_gradient_physical_hz_per_mm(line)
                line["data"] = self.scale_gradient_data(raw_data, raw_units)
                line["units"] = self.get_gradient_display_units(raw_units)

    def classify_gradient_axis(self, line: dict) -> str | None:
        for candidate in (
            str(line.get("key", "")),
            str(line.get("label", "")),
            str(line.get("chanLabel", "")),
        ):
            normalized = re.sub(r"[^a-z0-9]", "", candidate.lower())
            if normalized in {"gx", "g1", "gradx", "gradientx"}:
                return "x"
            if normalized in {"gy", "g2", "grady", "gradienty"}:
                return "y"
            if normalized in {"gz", "g3", "gradz", "gradientz"}:
                return "z"
        return None

    def normalize_time_series(self, time: np.ndarray, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        time_array = np.asarray(time, dtype=float)
        data_array = np.asarray(data, dtype=float)

        if time_array.size <= 1:
            return time_array, data_array

        keep_indices = [0]
        for index in range(1, time_array.size):
            current_time = float(time_array[index])
            last_kept_index = keep_indices[-1]
            last_time = float(time_array[last_kept_index])

            if current_time > last_time:
                keep_indices.append(index)
            else:
                keep_indices[-1] = index

        keep_array = np.asarray(keep_indices, dtype=int)
        return time_array[keep_array], data_array[keep_array]

    def compute_gradient_slew_rate_profile(self, time: np.ndarray, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        norm_time, norm_data = self.normalize_time_series(time, data)
        if norm_time.size < 2 or norm_data.size < 2:
            return norm_time, np.zeros_like(norm_time, dtype=float)

        interval_slew = np.diff(norm_data) / np.diff(norm_time)
        slew_data = np.empty_like(norm_time, dtype=float)
        slew_data[:-1] = interval_slew
        slew_data[-1] = interval_slew[-1]
        return norm_time, slew_data

    def compute_gradient_trajectory(self, time: np.ndarray, data: np.ndarray) -> np.ndarray:
        norm_time, norm_data = self.normalize_time_series(time, data)
        trajectory = np.zeros_like(norm_data, dtype=float)
        if norm_time.size < 2 or norm_data.size < 2:
            return trajectory

        interval_area = 0.5 * (norm_data[:-1] + norm_data[1:]) * np.diff(norm_time)
        trajectory[1:] = np.cumsum(interval_area)
        return trajectory

    def compute_gradient_duty_cycle(self, time: np.ndarray, data: np.ndarray) -> np.ndarray:
        norm_time, norm_data = self.normalize_time_series(time, data)
        duty_cycle = np.zeros_like(norm_data, dtype=float)
        if norm_time.size < 2 or norm_data.size < 2:
            return duty_cycle

        fake_interval = max(float(self.derivedSignalStartupPadding), 0.0)
        duty_time = np.concatenate(([norm_time[0] - fake_interval], norm_time))
        duty_data = np.concatenate(([0.0], norm_data))

        dt = np.diff(duty_time)
        active_intervals = (np.abs(duty_data[:-1]) > 1e-12) * dt
        cumulative_active = np.concatenate(([0.0], np.cumsum(active_intervals)))
        elapsed = norm_time - duty_time[0]
        valid = elapsed > 0
        duty_cycle[valid] = 100.0 * cumulative_active[1:][valid] / elapsed[valid]
        return duty_cycle

    def zero_trajectory_to_reference(self, time: np.ndarray, trajectory: np.ndarray) -> np.ndarray:
        if self.trajectoryZeroReferenceTime is None or time.size == 0 or trajectory.size == 0:
            return trajectory

        reference_value = float(
            np.interp(
                self.trajectoryZeroReferenceTime,
                np.asarray(time, dtype=float),
                np.asarray(trajectory, dtype=float),
                left=float(trajectory[0]),
                right=float(trajectory[-1]),
            ),
        )
        return np.asarray(trajectory, dtype=float) - reference_value

    def get_nco_channel_role(self, line: dict) -> tuple[str, str] | None:
        if line.get("type") != "NCO":
            return None

        nco_id = str(line.get("ind", "")).strip()
        key = str(line.get("key", "")).strip().lower()
        if key in {"am", "pw"} and nco_id:
            return nco_id, key

        for candidate in (str(line.get("label", "")), str(line.get("chanLabel", ""))):
            match = re.search(r"NCO_(\d+)_(am|pw)\b", candidate, flags=re.IGNORECASE)
            if match:
                return match.group(1), match.group(2).lower()

        return None

    def sample_step_series(
        self,
        source_time: np.ndarray,
        source_data: np.ndarray,
        query_time: np.ndarray,
    ) -> np.ndarray:
        if query_time.size == 0:
            return np.zeros(0, dtype=float)

        norm_time, norm_data = self.normalize_time_series(source_time, source_data)
        if norm_time.size == 0:
            return np.zeros_like(query_time, dtype=float)

        indices = np.searchsorted(norm_time, query_time, side="right") - 1
        indices = np.clip(indices, 0, norm_time.size - 1)
        sampled = norm_data[indices].astype(float, copy=True)
        sampled[query_time < norm_time[0]] = float(norm_data[0])
        return sampled

    def build_nco_power_derived_channels(self) -> list[list[dict]]:
        nco_sources: dict[str, dict[str, dict]] = {}
        for channel in self.channels:
            for line in channel:
                role = self.get_nco_channel_role(line)
                if role is None:
                    continue
                nco_id, key = role
                nco_sources.setdefault(nco_id, {})[key] = line

        derived_channels: list[list[dict]] = []
        for nco_id in sorted(nco_sources):
            sources = nco_sources[nco_id]
            if "am" not in sources or "pw" not in sources:
                continue

            am_line = sources["am"]
            pw_line = sources["pw"]
            am_time, am_data = self.normalize_time_series(
                np.asarray(am_line["t"], dtype=float),
                np.asarray(am_line["data"], dtype=float),
            )
            pw_time, pw_data = self.normalize_time_series(
                np.asarray(pw_line["t"], dtype=float),
                np.asarray(pw_line["data"], dtype=float),
            )
            merged_time = np.unique(np.concatenate((am_time, pw_time)))
            if merged_time.size == 0:
                continue

            sampled_am = self.sample_step_series(am_time, am_data, merged_time)
            sampled_pw = self.sample_step_series(pw_time, pw_data, merged_time)
            output_power = sampled_pw * np.square(sampled_am / 100.0)
            energy = np.zeros_like(output_power, dtype=float)
            if merged_time.size > 1:
                energy[1:] = np.cumsum(output_power[:-1] * np.diff(merged_time))

            average_power = np.zeros_like(output_power, dtype=float)
            average_padding = max(float(self.derivedSignalStartupPadding), 0.0)
            average_time = np.concatenate(([merged_time[0] - average_padding], merged_time))
            average_power_profile = np.concatenate(([0.0], output_power))
            average_energy = np.zeros_like(average_time, dtype=float)
            if average_time.size > 1:
                average_energy[1:] = np.cumsum(average_power_profile[:-1] * np.diff(average_time))
            elapsed = merged_time - average_time[0]
            valid = elapsed > 0
            average_power[valid] = average_energy[1:][valid] / elapsed[valid]

            derived_channels.extend(
                [
                    [
                        {
                            "chanLabel": f"NCO_{nco_id} Output Power",
                            "label": f"NCO_{nco_id}_pout",
                            "type": "nco_derived",
                            "ind": nco_id,
                            "key": "pout",
                            "plotType": "power",
                            "units": "W",
                            "t": merged_time,
                            "data": output_power,
                            "annotations": [],
                            "drawStyle": "step",
                            "show": False,
                        },
                    ],
                    [
                        {
                            "chanLabel": f"NCO_{nco_id} Energy",
                            "label": f"NCO_{nco_id}_energy",
                            "type": "nco_derived",
                            "ind": nco_id,
                            "key": "energy",
                            "plotType": "mag",
                            "units": "J",
                            "t": merged_time,
                            "data": energy,
                            "annotations": [],
                            "drawStyle": "line",
                            "show": False,
                        },
                    ],
                    [
                        {
                            "chanLabel": f"NCO_{nco_id} Average Power",
                            "label": f"NCO_{nco_id}_pavg",
                            "type": "nco_derived",
                            "ind": nco_id,
                            "key": "pavg",
                            "plotType": "power",
                            "units": "W",
                            "t": merged_time,
                            "data": average_power,
                            "annotations": [],
                            "drawStyle": "line",
                            "show": False,
                        },
                    ],
                ],
            )

        return derived_channels

    def apply_trajectory_zero_in_place(self) -> None:
        if not self.channels:
            return

        for channel, plot in zip(self.channels, self.plots, strict=False):
            if not channel or channel[0].get("chanLabel") != "Gradient Trajectory":
                continue

            for line_index, line in enumerate(channel):
                raw_trajectory = np.asarray(line.get("raw_data", line.get("data", [])), dtype=float)
                time = np.asarray(line.get("t", []), dtype=float)
                zeroed_trajectory = self.zero_trajectory_to_reference(time, raw_trajectory)
                line["data"] = zeroed_trajectory

                if line_index < len(plot.managed_curves):
                    plot.update_managed_curve(line_index, time, zeroed_trajectory)

    def build_gradient_derived_channels(self) -> list[list[dict]]:
        gradient_axes: dict[str, dict] = {}
        for channel in self.channels:
            for line in channel:
                if line.get("type") != "grads":
                    continue
                axis = self.classify_gradient_axis(line)
                if axis is not None and axis not in gradient_axes:
                    gradient_axes[axis] = line

        if not gradient_axes:
            return []

        derived_channels: list[list[dict]] = []
        axis_meta = {"x": ("Gx", "g"), "y": ("Gy", "r"), "z": ("Gz", "b")}

        slew_channel: list[dict] = []
        trajectory_channel: list[dict] = []
        duty_cycle_channel: list[dict] = []

        for axis in ("x", "y", "z"):
            if axis not in gradient_axes:
                continue

            source_line = gradient_axes[axis]
            _, pen = axis_meta[axis]
            time = np.asarray(source_line["t"], dtype=float)
            display_data = np.asarray(source_line["data"], dtype=float)
            display_units = str(source_line.get("units", "")).strip()
            physical_hz_per_mm = source_line.get("physical_hz_per_mm")

            if physical_hz_per_mm is not None:
                physical_hz_per_mm = np.asarray(physical_hz_per_mm, dtype=float)
                _, slew_hz_per_mm = self.compute_gradient_slew_rate_profile(time, physical_hz_per_mm)
                if self.displayGradientsInMtPerM:
                    slew_time, _ = self.normalize_time_series(time, physical_hz_per_mm)
                    slew_data = self.hz_per_mm_to_t_per_m(slew_hz_per_mm)
                    slew_units = "T/m/s"
                else:
                    slew_time, _ = self.normalize_time_series(time, physical_hz_per_mm)
                    slew_data = slew_hz_per_mm
                    slew_units = "Hz/mm/s"

                traj_time, traj_source = self.normalize_time_series(time, physical_hz_per_mm)
                traj_data = self.compute_gradient_trajectory(traj_time, traj_source)
                traj_units = "cycles/mm"
            else:
                time, data = self.normalize_time_series(time, display_data)
                slew_time, slew_data = self.compute_gradient_slew_rate_profile(time, data)
                slew_units = f"{display_units}/s" if display_units else "a.u./s"
                traj_time = time
                traj_data = self.compute_gradient_trajectory(time, data)
                traj_units = f"{display_units}*s" if display_units else "a.u.*s"

            traj_data = self.zero_trajectory_to_reference(traj_time, traj_data)
            duty_cycle_time, duty_cycle_source = self.normalize_time_series(time, display_data)
            duty_cycle_data = self.compute_gradient_duty_cycle(duty_cycle_time, duty_cycle_source)

            slew_channel.append(
                {
                    "chanLabel": "Gradient Slew Rate",
                    "label": f"S{axis}",
                    "type": "grads_derived",
                    "ind": source_line.get("ind", axis),
                    "key": f"S{axis}",
                    "plotType": "mag",
                    "units": slew_units,
                    "t": slew_time,
                    "data": slew_data,
                    "annotations": [],
                    "pen": pen,
                    "drawStyle": "step",
                    "show": False,
                },
            )
            trajectory_channel.append(
                {
                    "chanLabel": "Gradient Trajectory",
                    "label": f"T{axis}",
                    "type": "grads_derived",
                    "ind": source_line.get("ind", axis),
                    "key": f"T{axis}",
                    "plotType": "mag",
                    "units": traj_units,
                    "t": traj_time,
                    "raw_data": traj_data.copy(),
                    "data": traj_data,
                    "annotations": [],
                    "pen": pen,
                    "drawStyle": "line",
                    "show": False,
                },
            )
            duty_cycle_channel.append(
                {
                    "chanLabel": "Gradient Duty Cycle",
                    "label": f"D{axis}",
                    "type": "grads_derived",
                    "ind": source_line.get("ind", axis),
                    "key": f"D{axis}",
                    "plotType": "mag",
                    "units": "%",
                    "t": duty_cycle_time,
                    "data": duty_cycle_data,
                    "annotations": [],
                    "pen": pen,
                    "drawStyle": "step",
                    "show": False,
                },
            )

        if slew_channel:
            derived_channels.append(slew_channel)
        if trajectory_channel:
            derived_channels.append(trajectory_channel)
        if duty_cycle_channel:
            derived_channels.append(duty_cycle_channel)

        return derived_channels

    def slice_curve_to_range(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        x_min: float,
        x_max: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if x_data.size == 0 or y_data.size == 0:
            return x_data, y_data

        visible_mask = (x_data >= x_min) & (x_data <= x_max)
        visible_indices = np.flatnonzero(visible_mask)

        if visible_indices.size == 0:
            right_index = int(np.searchsorted(x_data, x_min, side="left"))
            candidate_indices = {min(max(right_index - 1, 0), x_data.size - 1), min(right_index, x_data.size - 1)}
            selected_indices = np.array(sorted(candidate_indices), dtype=int)
            return x_data[selected_indices], y_data[selected_indices]

        start_index = max(int(visible_indices[0]) - 1, 0)
        end_index = min(int(visible_indices[-1]) + 1, x_data.size - 1)
        selected_indices = np.arange(start_index, end_index + 1)
        return x_data[selected_indices], y_data[selected_indices]

    def simplify_curve_indices(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        x_scale: float,
        y_scale: float,
        tolerance_px: float,
    ) -> np.ndarray:
        point_count = x_data.size
        if point_count <= 2:
            return np.arange(point_count, dtype=int)

        x_screen = x_data * x_scale
        y_screen = y_data * y_scale
        keep_mask = np.zeros(point_count, dtype=bool)
        keep_mask[0] = True
        keep_mask[-1] = True
        stack: list[tuple[int, int]] = [(0, point_count - 1)]
        tolerance_sq = tolerance_px * tolerance_px

        while stack:
            start_index, end_index = stack.pop()
            if end_index <= start_index + 1:
                continue

            start_point = np.array((x_screen[start_index], y_screen[start_index]))
            end_point = np.array((x_screen[end_index], y_screen[end_index]))
            segment = end_point - start_point
            segment_length_sq = float(np.dot(segment, segment))

            interior_slice = slice(start_index + 1, end_index)
            points = np.column_stack((x_screen[interior_slice], y_screen[interior_slice]))
            if points.size == 0:
                continue

            if segment_length_sq <= 1e-12:
                distances_sq = np.sum((points - start_point) ** 2, axis=1)
            else:
                projection = np.clip(np.dot(points - start_point, segment) / segment_length_sq, 0.0, 1.0)
                closest_points = start_point + np.outer(projection, segment)
                distances_sq = np.sum((points - closest_points) ** 2, axis=1)

            max_offset = int(np.argmax(distances_sq))
            max_distance_sq = float(distances_sq[max_offset])
            if max_distance_sq > tolerance_sq:
                split_index = start_index + 1 + max_offset
                keep_mask[split_index] = True
                stack.append((start_index, split_index))
                stack.append((split_index, end_index))

        return np.flatnonzero(keep_mask)

    def prebin_curve_to_viewport(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        viewport_width: int,
        x_min: float,
        x_max: float,
        *,
        bins_per_pixel: int = 2,
    ) -> tuple[np.ndarray, np.ndarray]:
        if x_data.size <= 2:
            return x_data, y_data

        bin_count = max(int(viewport_width * bins_per_pixel), 1)
        if x_data.size <= bin_count * 4:
            return x_data, y_data

        x_span = max(abs(x_max - x_min), 1e-12)
        normalized = (x_data - x_min) / x_span
        bin_indices = np.clip((normalized * bin_count).astype(int), 0, bin_count - 1)

        kept_indices: list[int] = [0]
        start = 0
        while start < x_data.size:
            current_bin = int(bin_indices[start])
            end = start + 1
            while end < x_data.size and int(bin_indices[end]) == current_bin:
                end += 1

            segment = slice(start, end)
            segment_y = y_data[segment]
            if segment_y.size > 0:
                local_min = start + int(np.argmin(segment_y))
                local_max = start + int(np.argmax(segment_y))
                kept_indices.extend((local_min, local_max, end - 1))

            start = end

        kept_indices.append(x_data.size - 1)
        unique_indices = np.unique(np.asarray(kept_indices, dtype=int))
        return x_data[unique_indices], y_data[unique_indices]

    def downsample_curve_to_viewport(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        viewport_width: int,
        viewport_height: int,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        *,
        max_point_factor: float = 6.0,
        min_points: int = 3000,
    ) -> tuple[np.ndarray, np.ndarray]:
        if x_data.size <= 2:
            return x_data, y_data

        if not np.all(np.isfinite(x_data)) or not np.all(np.isfinite(y_data)):
            return x_data, y_data

        x_span = max(abs(x_max - x_min), 1e-12)
        y_span = max(abs(y_max - y_min), 1e-12)
        x_scale = max(viewport_width - 1, 1) / x_span
        y_scale = max(viewport_height - 1, 1) / y_span
        max_points = max(int(viewport_width * max_point_factor), min_points)

        if x_data.size <= max_points:
            return x_data, y_data

        prebin_target = max(max_points * 4, viewport_width * 8)
        if x_data.size > prebin_target:
            x_data, y_data = self.prebin_curve_to_viewport(
                x_data,
                y_data,
                viewport_width,
                x_min,
                x_max,
            )
            if x_data.size <= max_points:
                return x_data, y_data

        low_tolerance = 0.0
        high_tolerance = 0.5
        kept_indices = self.simplify_curve_indices(x_data, y_data, x_scale, y_scale, high_tolerance)

        while kept_indices.size > max_points and high_tolerance < 64.0:
            low_tolerance = high_tolerance
            high_tolerance *= 2.0
            kept_indices = self.simplify_curve_indices(x_data, y_data, x_scale, y_scale, high_tolerance)

        for _ in range(16):
            if kept_indices.size <= max_points:
                break
            mid_tolerance = (low_tolerance + high_tolerance) * 0.5
            kept_indices = self.simplify_curve_indices(x_data, y_data, x_scale, y_scale, mid_tolerance)
            if kept_indices.size > max_points:
                low_tolerance = mid_tolerance
            else:
                high_tolerance = mid_tolerance

        if kept_indices.size > max_points:
            sample_indices = np.linspace(0, kept_indices.size - 1, num=max_points, dtype=int)
            kept_indices = kept_indices[np.unique(sample_indices)]

        return x_data[kept_indices], y_data[kept_indices]
