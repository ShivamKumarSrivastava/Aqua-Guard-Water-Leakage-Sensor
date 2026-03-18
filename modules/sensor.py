"""
modules/sensor.py
-----------------
Sensor class + dataset generator.
"""

import random
import pandas as pd

ZONES = ["Residential_A", "Industrial_B", "Commercial_C", "Park_Zone"]


class Sensor:
    def __init__(self, sensor_id, location, flow_rate, baseline, pressure):
        self.sensor_id  = sensor_id
        self.location   = location
        self.flow_rate  = flow_rate
        self.baseline   = baseline
        self.pressure   = pressure

    def check_status(self) -> str:
        if self.flow_rate == 0 and self.baseline > 10:
            return "Major_Burst"
        if self.pressure > 80:
            return "High_Pressure"
        if self.flow_rate < 0.5 * self.baseline and self.flow_rate != 0:
            return "Minor_Leak"
        return "Normal"


def generate_sensors(n: int = 300) -> list[Sensor]:
    """Create n random Sensor objects."""
    sensors = []
    for i in range(n):
        baseline   = random.uniform(10, 200)
        flow_rate  = random.uniform(5, 250)
        pressure   = random.uniform(20, 90)

        if random.random() < 0.05:
            flow_rate = 0
        if random.random() < 0.05:
            pressure = random.uniform(90, 150)

        sensors.append(Sensor(
            sensor_id = f"S_{i:03}",
            location  = random.choice(ZONES),
            flow_rate = flow_rate,
            baseline  = baseline,
            pressure  = pressure,
        ))
    return sensors


def sensors_to_df(sensors: list[Sensor]) -> pd.DataFrame:
    """Convert a list of Sensor objects to a labelled DataFrame."""
    rows = []
    for s in sensors:
        rows.append({
            "sensor_id":      s.sensor_id,
            "location_zone":  s.location,
            "flow_rate_lpm":  s.flow_rate,
            "baseline_mean":  s.baseline,
            "pressure_psi":   s.pressure,
            "active_label":   s.check_status(),
        })
    return pd.DataFrame(rows)


def generate_dataframe(n: int = 300) -> pd.DataFrame:
    """One-shot helper: generate sensors and return a DataFrame."""
    return sensors_to_df(generate_sensors(n))
