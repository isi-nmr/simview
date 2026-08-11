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

    def get_gradient_display_mode(self) -> str:
        mode = str(getattr(self, "gradientDisplayUnits", "hz_per_mm")).lower()
        if mode in {"percent", "hz_per_mm", "mt_per_m"}:
            return mode
        return "hz_per_mm"

    def get_gradient_display_units(self, raw_units: str | None) -> str:
        if self.gradientCalibrationHzPerMm > 0 and self.is_percent_gradient_units(raw_units):
            mode = self.get_gradient_display_mode()
            if mode == "percent":
                return "%"
            if mode == "mt_per_m":
                return "mT/m"
            return "Hz/mm"
        return (raw_units or "").strip() or "%"

    def scale_gradient_data(self, data: np.ndarray, raw_units: str | None) -> np.ndarray:
        scaled = np.asarray(data, dtype=float)
        if self.gradientCalibrationHzPerMm > 0 and self.is_percent_gradient_units(raw_units):
            mode = self.get_gradient_display_mode()
            if mode == "percent":
                return scaled
            scaled_hz_per_mm = scaled * (self.gradientCalibrationHzPerMm / 100.0)
            if mode == "mt_per_m":
                return self.hz_per_mm_to_mt_per_m(scaled_hz_per_mm)
            return scaled_hz_per_mm
        return scaled

    def get_gradient_physical_hz_per_mm(self, line: dict) -> np.ndarray | None:
        if self.gradientCalibrationHzPerMm <= 0:
            return None

        if self.classify_gradient_axis(line) is None:
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
                if line["physical_hz_per_mm"] is None:
                    line["data"] = raw_data
                    line["units"] = raw_units.strip() or "%"
                else:
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

    def apply_trajectory_refocuses(self, time: np.ndarray, trajectory: np.ndarray) -> np.ndarray:
        refocus_times = sorted(
            {
                float(time_value)
                for time_value in getattr(self, "trajectoryRefocusTimes", [])
                if np.isfinite(float(time_value))
            },
        )
        if time.size == 0 or trajectory.size == 0:
            return trajectory

        time_array = np.asarray(time, dtype=float)
        trajectory_array = np.asarray(trajectory, dtype=float)
        reset_times = self.get_trajectory_excitation_times(time_array)
        if not refocus_times and reset_times.size == 0:
            return trajectory
        anchor_time = getattr(self, "trajectoryZeroReferenceTime", None)
        if anchor_time is None:
            anchor_time = float(time_array[0])
        anchor_time = float(np.clip(anchor_time, time_array[0], time_array[-1]))

        in_range_refocuses = [
            value for value in refocus_times if float(time_array[0]) <= value <= float(time_array[-1])
        ]
        integration_times = np.unique(
            np.concatenate((time_array, np.asarray(in_range_refocuses, dtype=float), reset_times, [anchor_time])),
        )
        integration_values = np.interp(integration_times, time_array, trajectory_array)

        # A refocus changes the sign of every subsequent gradient contribution.
        # Integrating those signed increments from the original trajectory zero
        # keeps the coherence path continuous instead of merely reflecting its
        # accumulated value at each refocus event.
        interval_midpoints = 0.5 * (integration_times[:-1] + integration_times[1:])
        refocus_array = np.asarray(in_range_refocuses, dtype=float)
        anchor_flip_count = int(np.searchsorted(refocus_array, anchor_time, side="right"))
        midpoint_flip_counts = np.searchsorted(refocus_array, interval_midpoints, side="right")
        crossed_refocuses = np.abs(midpoint_flip_counts - anchor_flip_count)
        interval_signs = np.where(crossed_refocuses % 2 == 0, 1.0, -1.0)

        signed_increments = np.diff(integration_values) * interval_signs
        integrated = np.concatenate(([0.0], np.cumsum(signed_increments)))
        anchor_index = int(np.searchsorted(integration_times, anchor_time))
        anchor_value = float(np.interp(anchor_time, time_array, trajectory_array))
        integrated += anchor_value - integrated[anchor_index]

        # Every excitation begins a new repetition block.  Its coherence
        # trajectory has a new origin, so neither residual moment nor refocus
        # parity is allowed to carry through into the next TR.
        for reset_index, reset_time in enumerate(reset_times):
            start_index = int(np.searchsorted(integration_times, reset_time))
            end_time = reset_times[reset_index + 1] if reset_index + 1 < reset_times.size else integration_times[-1]
            end_index = int(np.searchsorted(integration_times, end_time, side="right") - 1)
            if start_index > end_index:
                continue
            integrated[start_index] = 0.0
            for index in range(start_index, end_index):
                midpoint = 0.5 * (integration_times[index] + integration_times[index + 1])
                flips_in_block = np.searchsorted(refocus_array, midpoint, side="right") - np.searchsorted(
                    refocus_array, reset_time, side="right",
                )
                sign = -1.0 if flips_in_block % 2 else 1.0
                integrated[index + 1] = integrated[index] + sign * (integration_values[index + 1] - integration_values[index])
        return np.interp(time_array, integration_times, integrated)

    def get_trajectory_excitation_times(self, time: np.ndarray) -> np.ndarray:
        """Find calibrated 90-degree RF foci that start independent TR blocks."""
        if not hasattr(self, "detect_rf_pulse_descriptors"):
            return np.asarray([], dtype=float)
        calibrations = [
            item for item in self.__dict__.get("rfPulseCalibrations", [])
            if abs(float(item.get("flip_angle", 0.0)) - 90.0) <= 10.0
        ]
        if not calibrations:
            return np.asarray([], dtype=float)
        excitations = [
            float(pulse["focus"])
            for pulse in self.detect_rf_pulse_descriptors()
            if any(
                np.isclose(float(pulse["duration"]), float(calibration["duration"]), rtol=0.03, atol=2e-6)
                for calibration in calibrations
            )
        ]
        if not excitations:
            return np.asarray([], dtype=float)
        return np.asarray(sorted({value for value in excitations if time[0] <= value <= time[-1]}), dtype=float)

    def apply_trajectory_display_transforms(self, time: np.ndarray, trajectory: np.ndarray) -> np.ndarray:
        zeroed = self.zero_trajectory_to_reference(time, trajectory)
        return self.apply_trajectory_refocuses(time, zeroed)

    def get_trajectory_display_profile(self, time: np.ndarray, raw_trajectory: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        source_time = np.asarray(time, dtype=float)
        source_data = np.asarray(raw_trajectory, dtype=float)
        if source_time.size == 0 or source_data.size == 0:
            return source_time, source_data
        knot_parts = [
            source_time,
            np.asarray(getattr(self, "trajectoryRefocusTimes", []), dtype=float),
            self.get_trajectory_excitation_times(source_time),
        ]
        if self.trajectoryZeroReferenceTime is not None:
            knot_parts.append(np.asarray([self.trajectoryZeroReferenceTime], dtype=float))
        display_time = np.unique(np.concatenate([part for part in knot_parts if part.size]))
        display_time = display_time[(display_time >= source_time[0]) & (display_time <= source_time[-1])]
        display_raw = np.interp(display_time, source_time, source_data)
        return display_time, self.apply_trajectory_display_transforms(display_time, display_raw)

    def compute_coherence_order_profile(self, time: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        time_array = np.asarray(time, dtype=float)
        if time_array.size == 0:
            return time_array, np.asarray([], dtype=float)

        start_time = float(time_array[0])
        end_time = float(time_array[-1])
        anchor_time = getattr(self, "trajectoryZeroReferenceTime", None)
        if anchor_time is None:
            anchor_time = start_time
        anchor_time = float(np.clip(anchor_time, start_time, end_time))
        excitation_time = start_time
        if hasattr(self, "detect_rf_pulse_descriptors"):
            descriptors = self.detect_rf_pulse_descriptors()
            if descriptors:
                excitation_time = float(np.clip(float(descriptors[0]["focus"]), start_time, end_time))
        flip_times = sorted(
            {
                float(value)
                for value in getattr(self, "trajectoryRefocusTimes", [])
                if np.isfinite(float(value)) and start_time <= float(value) <= end_time
            },
        )

        profile_time = np.unique(np.asarray([start_time, excitation_time, *flip_times, end_time], dtype=float))
        order_at_start = -1.0
        if sum(start_time < value <= anchor_time for value in flip_times) % 2:
            order_at_start = 1.0

        coherence_order = np.full(profile_time.shape, order_at_start, dtype=float)
        coherence_order[profile_time < excitation_time] = 0.0
        for index, time_value in enumerate(profile_time):
            flips_since_start = sum(value <= time_value for value in flip_times)
            if flips_since_start % 2:
                coherence_order[index] *= -1.0
        return profile_time, coherence_order

    def compute_imperfect_refocus_pathway_weights(self, time: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return population weights for p-, p+, and p0 through the RF train.

        A nominal refocusing pulse is modelled as a rotation of the coherence
        basis.  At 180 degrees this reduces to the familiar p- <-> p+ swap.
        Away from 180 degrees it retains transverse coherence and transfers a
        portion through p0, making the otherwise hidden pathways inspectable.
        """
        profile_time, _order = self.compute_coherence_order_profile(time)
        if profile_time.size == 0:
            return profile_time, np.empty((3, 0), dtype=float)

        excitation_time = profile_time[0]
        if hasattr(self, "detect_rf_pulse_descriptors"):
            descriptors = self.detect_rf_pulse_descriptors()
            if descriptors:
                excitation_time = float(descriptors[0]["focus"])
        flip_times = {
            float(value) for value in getattr(self, "trajectoryRefocusTimes", []) if np.isfinite(float(value))
        }
        theta = np.deg2rad(float(self.__dict__.get("trajectoryRefocusFlipAngleDegrees", 180.0)))
        transfer = float(np.sin(theta / 2.0) ** 4)
        retained = float(np.cos(theta / 2.0) ** 4)
        transverse_to_zero = float(0.5 * np.sin(theta) ** 2)
        zero_to_transverse = float(0.5 * np.sin(theta) ** 2)
        zero_retained = float(np.cos(theta) ** 2)

        # Rows are p-, p+, p0.  Longitudinal coherence is present before the
        # excitation pulse; the selected transverse pathway begins at p-.
        weights = np.zeros((3, profile_time.size), dtype=float)
        state = (
            np.asarray([1.0, 0.0, 0.0], dtype=float)
            if excitation_time <= profile_time[0]
            else np.asarray([0.0, 0.0, 1.0], dtype=float)
        )
        for index, time_value in enumerate(profile_time):
            if time_value >= excitation_time and index > 0 and profile_time[index - 1] < excitation_time:
                state = np.asarray([1.0, 0.0, 0.0], dtype=float)
            if time_value in flip_times:
                p_minus, p_plus, p_zero = state
                state = np.asarray([
                    retained * p_minus + transfer * p_plus + zero_to_transverse * p_zero,
                    transfer * p_minus + retained * p_plus + zero_to_transverse * p_zero,
                    transverse_to_zero * (p_minus + p_plus) + zero_retained * p_zero,
                ], dtype=float)
            weights[:, index] = state
        return profile_time, weights

    def compute_imperfect_coherence_branches(
        self,
        time: np.ndarray,
        max_branches: int = 12,
    ) -> tuple[np.ndarray, list[dict[str, object]]]:
        """Build the strongest explicit coherence histories through the RF train.

        The order values (p-, p+, p0) are states, not pathways.  A pathway is
        a particular succession of those states, so branches are retained by
        RF history and pruned only by their relative weight.
        """
        profile_time, _weights = self.compute_imperfect_refocus_pathway_weights(time)
        if profile_time.size == 0:
            return profile_time, []
        excitation_time = profile_time[0]
        if hasattr(self, "detect_rf_pulse_descriptors"):
            descriptors = self.detect_rf_pulse_descriptors()
            if descriptors:
                excitation_time = float(descriptors[0]["focus"])
        flip_times = {
            float(value) for value in getattr(self, "trajectoryRefocusTimes", []) if np.isfinite(float(value))
        }
        theta = np.deg2rad(float(self.__dict__.get("trajectoryRefocusFlipAngleDegrees", 180.0)))
        transitions = {
            -1: ((1, np.sin(theta / 2.0) ** 4), (-1, np.cos(theta / 2.0) ** 4), (0, 0.5 * np.sin(theta) ** 2)),
            1: ((-1, np.sin(theta / 2.0) ** 4), (1, np.cos(theta / 2.0) ** 4), (0, 0.5 * np.sin(theta) ** 2)),
            0: ((-1, 0.5 * np.sin(theta) ** 2), (1, 0.5 * np.sin(theta) ** 2), (0, np.cos(theta) ** 2)),
        }
        name = {-1: "p−", 0: "p0", 1: "p+"}
        branches: list[dict[str, object]] = [{"order": 0, "weight": 1.0, "history": [], "data": []}]
        for index, time_value in enumerate(profile_time):
            if time_value >= excitation_time and (index == 0 or profile_time[index - 1] < excitation_time):
                branches = [{"order": -1, "weight": 1.0, "history": ["p−"], "data": [0.0] * index}]
            if time_value in flip_times:
                next_branches: list[dict[str, object]] = []
                for branch in branches:
                    for destination, factor in transitions[int(branch["order"])]:
                        if factor <= 1e-14:
                            continue
                        next_branches.append({
                            "order": destination,
                            "weight": float(branch["weight"]) * float(factor),
                            "history": [*branch["history"], name[destination]],
                            "data": [*branch["data"], destination],
                        })
                branches = sorted(next_branches, key=lambda item: float(item["weight"]), reverse=True)[:max_branches]
            else:
                for branch in branches:
                    branch["data"].append(int(branch["order"]))

        result: list[dict[str, object]] = []
        for index, branch in enumerate(branches, start=1):
            history = " → ".join(str(value) for value in branch["history"])
            result.append({
                "label": f"Path {index} ({float(branch['weight']):.3g}) · {history}",
                "weight": float(branch["weight"]),
                "data": np.asarray(branch["data"], dtype=float),
            })
        return profile_time, result

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
                source_time = np.asarray(line.get("raw_t", line.get("t", [])), dtype=float)
                time, display_trajectory = self.get_trajectory_display_profile(source_time, raw_trajectory)
                line["raw_t"] = source_time
                line["t"] = time
                line["data"] = display_trajectory

                if line_index < len(plot.managed_curves):
                    plot.update_managed_curve(line_index, time, display_trajectory)

        self.update_coherence_order_in_place()
        self.update_trajectory_residual_in_place()

        if hasattr(self, "refresh_trajectory_flip_markers"):
            self.refresh_trajectory_flip_markers()

    def update_coherence_order_in_place(self) -> None:
        for channel, plot in zip(self.channels, self.plots, strict=False):
            if not channel or channel[0].get("chanLabel") not in {
                "Coherence Order", "Candidate Coherence Pathways", "Imperfect RF Pathway Weights",
            }:
                continue
            line = channel[0]
            source_time = np.asarray(line.get("source_time", line.get("t", [])), dtype=float)
            profile_time, coherence_order = self.compute_coherence_order_profile(source_time)
            _weight_time, pathway_weights = self.compute_imperfect_refocus_pathway_weights(source_time)
            _branch_time, branches = self.compute_imperfect_coherence_branches(source_time)
            for line_index, path_line in enumerate(channel):
                path_line["t"] = profile_time
                key = str(path_line.get("key", ""))
                if channel[0].get("chanLabel") == "Imperfect RF Pathway Weights":
                    weight_index = {"p-": 0, "p+": 1, "p0": 2}.get(key, 0)
                    path_line["data"] = pathway_weights[weight_index]
                elif channel[0].get("chanLabel") == "Candidate Coherence Pathways":
                    if line_index >= len(branches):
                        continue
                    path_line["data"] = np.asarray(branches[line_index]["data"], dtype=float)
                    path_line["label"] = str(branches[line_index]["label"])
                else:
                    path_line["data"] = (
                        -coherence_order if key == "p+" else np.zeros_like(coherence_order) if key == "p0" else coherence_order
                    )
                if line_index >= len(plot.managed_curves):
                    continue
                step_time = np.repeat(profile_time, 2)[1:]
                step_order = np.repeat(path_line["data"], 2)[:-1]
                plot.update_managed_curve(line_index, step_time, step_order)

    def compute_trajectory_residual(self, trajectory_channel: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        if not trajectory_channel:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)
        knot_parts = [np.asarray(line.get("t", []), dtype=float) for line in trajectory_channel]
        knot_parts.append(np.asarray(getattr(self, "trajectoryRefocusTimes", []), dtype=float))
        if self.trajectoryZeroReferenceTime is not None:
            knot_parts.append(np.asarray([self.trajectoryZeroReferenceTime], dtype=float))
        time = np.unique(np.concatenate([part for part in knot_parts if part.size])) if any(
            part.size for part in knot_parts
        ) else np.asarray([], dtype=float)
        components: list[np.ndarray] = []
        for line in trajectory_channel:
            line_time = np.asarray(line.get("t", []), dtype=float)
            line_data = np.asarray(line.get("data", line.get("raw_data", [])), dtype=float)
            if line_time.size and line_data.size:
                components.append(np.interp(time, line_time, line_data))
        if not components:
            return time, np.zeros_like(time)

        # |K| is not linear between two linearly interpolated vector samples.
        # Insert each interval's analytic closest approach to the origin so the
        # displayed magnitude reaches the same minima as the Kx/Ky/Kz curves.
        vectors = np.vstack(components).T
        interval_duration = np.diff(time)
        deltas = np.diff(vectors, axis=0)
        denominators = np.sum(deltas * deltas, axis=1)
        fractions = np.divide(
            -np.sum(vectors[:-1] * deltas, axis=1),
            denominators,
            out=np.zeros_like(denominators),
            where=denominators > 1e-30,
        )
        interior = (fractions > 1e-12) & (fractions < 1.0 - 1e-12) & (interval_duration > 0)
        knot_times = [time]
        if np.any(interior):
            knot_times.append(time[:-1][interior] + fractions[interior] * interval_duration[interior])
        for component_index in range(vectors.shape[1]):
            component_delta = deltas[:, component_index]
            component_fraction = np.divide(
                -vectors[:-1, component_index],
                component_delta,
                out=np.zeros_like(component_delta),
                where=np.abs(component_delta) > 1e-30,
            )
            component_interior = (
                (component_fraction > 1e-12)
                & (component_fraction < 1.0 - 1e-12)
                & (interval_duration > 0)
            )
            if np.any(component_interior):
                knot_times.append(
                    time[:-1][component_interior] + component_fraction[component_interior] * interval_duration[component_interior],
                )
        if len(knot_times) > 1:
            time = np.unique(np.concatenate(knot_times))
            vectors = np.vstack([np.interp(time, np.asarray(line.get("t", []), dtype=float), np.asarray(line.get("data", line.get("raw_data", [])), dtype=float)) for line in trajectory_channel]).T
        return time, np.linalg.norm(vectors, axis=1)

    def update_trajectory_residual_in_place(self) -> None:
        trajectory = next(
            (channel for channel in self.channels if channel and channel[0].get("chanLabel") == "Gradient Trajectory"),
            [],
        )
        time, residual = self.compute_trajectory_residual(trajectory)
        for channel, plot in zip(self.channels, self.plots, strict=False):
            if channel and channel[0].get("chanLabel") == "Gradient Trajectory Residual":
                channel[0]["t"] = time
                channel[0]["data"] = residual
                if plot.managed_curves:
                    plot.update_managed_curve(0, time, residual)
                    plot.managed_curves[0]["item"].setData(time, residual)
                    # The residual gains interpolation knots at zero/flip times.
                    # Refresh now so a visible plot cannot retain its pre-zeroed
                    # downsampled rendering until the next view interaction.
                    if hasattr(plot, "refresh_visible_curves"):
                        plot.refresh_visible_curves()

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
        coherence_source_time: np.ndarray | None = None

        for axis in ("x", "y", "z"):
            if axis not in gradient_axes:
                continue

            source_line = gradient_axes[axis]
            _, pen = axis_meta[axis]
            time = np.asarray(source_line["t"], dtype=float)
            if coherence_source_time is None and time.size > 0:
                # Coherence order changes only at RF flips, so retaining the
                # complete (potentially multi-million-sample) gradient clock is
                # unnecessary. The profile builder inserts all flip times.
                coherence_source_time = np.asarray([time[0], time[-1]], dtype=float)
            display_data = np.asarray(source_line["data"], dtype=float)
            display_units = str(source_line.get("units", "")).strip()
            physical_hz_per_mm = source_line.get("physical_hz_per_mm")

            if physical_hz_per_mm is not None:
                physical_hz_per_mm = np.asarray(physical_hz_per_mm, dtype=float)
                _, slew_hz_per_mm = self.compute_gradient_slew_rate_profile(time, physical_hz_per_mm)
                if self.get_gradient_display_mode() == "mt_per_m":
                    slew_time, _ = self.normalize_time_series(time, physical_hz_per_mm)
                    slew_data = self.hz_per_mm_to_t_per_m(slew_hz_per_mm)
                    slew_units = "T/m/s"
                else:
                    slew_time, _ = self.normalize_time_series(time, physical_hz_per_mm)
                    slew_data = slew_hz_per_mm
                    slew_units = "Hz/mm/s"

                traj_time, traj_source = self.normalize_time_series(time, physical_hz_per_mm)
                raw_traj_data = self.compute_gradient_trajectory(traj_time, traj_source)
                traj_units = "cycles/mm"
            else:
                time, data = self.normalize_time_series(time, display_data)
                slew_time, slew_data = self.compute_gradient_slew_rate_profile(time, data)
                slew_units = f"{display_units}/s" if display_units else "a.u./s"
                traj_time = time
                raw_traj_data = self.compute_gradient_trajectory(time, data)
                traj_units = f"{display_units}*s" if display_units else "a.u.*s"

            display_traj_time, traj_data = self.get_trajectory_display_profile(traj_time, raw_traj_data)
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
                    "raw_t": traj_time.copy(),
                    "t": display_traj_time,
                    "raw_data": raw_traj_data.copy(),
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
            residual_time, residual_data = self.compute_trajectory_residual(trajectory_channel)
            derived_channels.append([{
                "chanLabel": "Gradient Trajectory Residual", "label": "|K|", "type": "grads_derived",
                "ind": "residual", "key": "Kmag", "plotType": "mag", "units": trajectory_channel[0]["units"],
                "t": residual_time, "data": residual_data, "annotations": [], "pen": "c", "drawStyle": "line", "show": False,
            }])
        if coherence_source_time is not None:
            coherence_time, coherence_order = self.compute_coherence_order_profile(coherence_source_time)
            derived_channels.append(
                [
                    {
                        "chanLabel": "Coherence Order",
                        "label": "p(t)",
                        "type": "coherence_order",
                        "ind": "p",
                        "key": "p",
                        "plotType": "mag",
                        "units": "",
                        "source_time": coherence_source_time.copy(),
                        "t": coherence_time,
                        "data": coherence_order,
                        "annotations": [],
                        "pen": "m",
                        "drawStyle": "step",
                        "show": False,
                    },
                ],
            )
            pathway_time, pathway_weights = self.compute_imperfect_refocus_pathway_weights(coherence_source_time)
            derived_channels.append(
                [
                    {
                        "chanLabel": "Imperfect RF Pathway Weights", "label": "p− weight", "type": "coherence_pathway",
                        "ind": "p-", "key": "p-", "plotType": "mag", "units": "relative weight",
                        "source_time": coherence_source_time.copy(), "t": pathway_time, "data": pathway_weights[0],
                        "annotations": [], "pen": "m", "drawStyle": "step", "show": False,
                    },
                    {
                        "chanLabel": "Imperfect RF Pathway Weights", "label": "p+ weight", "type": "coherence_pathway",
                        "ind": "p+", "key": "p+", "plotType": "mag", "units": "relative weight",
                        "source_time": coherence_source_time.copy(), "t": pathway_time, "data": pathway_weights[1],
                        "annotations": [], "pen": "c", "drawStyle": "step", "show": False,
                    },
                    {
                        "chanLabel": "Imperfect RF Pathway Weights", "label": "p0 weight", "type": "coherence_pathway",
                        "ind": "p0", "key": "p0", "plotType": "mag", "units": "relative weight",
                        "source_time": coherence_source_time.copy(), "t": pathway_time, "data": pathway_weights[2],
                        "annotations": [], "pen": "y", "drawStyle": "step", "show": False,
                    },
                ],
            )
            branch_time, branches = self.compute_imperfect_coherence_branches(coherence_source_time)
            branch_pens = ("m", "c", "y", "g", "r", "b")
            derived_channels.append([
                {
                    "chanLabel": "Candidate Coherence Pathways", "label": str(branch["label"]),
                    "type": "coherence_pathway", "ind": str(index), "key": f"path{index}",
                    "plotType": "mag", "units": "", "source_time": coherence_source_time.copy(),
                    "t": branch_time, "data": np.asarray(branch["data"], dtype=float), "annotations": [],
                    "pen": branch_pens[index % len(branch_pens)], "drawStyle": "step", "show": False,
                }
                for index, branch in enumerate(branches)
            ])
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
