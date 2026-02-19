"""스마트폰 배터리/온도 모니터링 모듈."""

import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class PhoneMonitor:
    """IP Webcam 앱의 센서 API를 통해 스마트폰 상태를 모니터링하는 클래스."""

    def __init__(self, config: dict):
        phone_cfg = config["camera"]["smartphone"]
        self.phone_ip = phone_cfg["phone_ip"]
        self.port = phone_cfg["port"]

        mon_cfg = config["camera"].get("monitoring", {})
        self.check_battery = mon_cfg.get("check_battery", True)
        self.battery_threshold = mon_cfg.get("battery_warning_threshold", 20)
        self.temp_threshold = mon_cfg.get("temp_warning_threshold", 45)

        self._last_battery: Optional[int] = None
        self._last_temp: Optional[float] = None
        self._is_charging: Optional[bool] = None

    @property
    def base_url(self) -> str:
        return f"http://{self.phone_ip}:{self.port}"

    def get_battery_status(self) -> dict:
        """배터리 상태를 조회한다."""
        try:
            resp = requests.get(
                f"{self.base_url}/sensors.json", timeout=3
            )
            if resp.status_code != 200:
                return {"available": False}

            data = resp.json()
            battery_data = data.get("battery_level", {})
            temp_data = data.get("battery_temp", {})

            battery_level = None
            if battery_data and "data" in battery_data:
                raw = battery_data["data"]
                if isinstance(raw, list) and len(raw) > 0:
                    battery_level = int(float(raw[0][1][0]))
                elif isinstance(raw, (int, float)):
                    battery_level = int(raw)

            battery_temp = None
            if temp_data and "data" in temp_data:
                raw = temp_data["data"]
                if isinstance(raw, list) and len(raw) > 0:
                    battery_temp = float(raw[0][1][0])
                elif isinstance(raw, (int, float)):
                    battery_temp = float(raw)

            self._last_battery = battery_level
            self._last_temp = battery_temp

            return {
                "available": True,
                "battery_level": battery_level,
                "battery_temp": battery_temp,
            }
        except (requests.RequestException, ValueError, KeyError) as e:
            logger.debug(f"Battery status check failed: {e}")
            return {"available": False}

    def check_status(self) -> dict:
        """스마트폰 상태를 확인하고 경고를 반환한다."""
        if not self.check_battery:
            return {"warnings": []}

        status = self.get_battery_status()
        warnings = []

        if not status["available"]:
            return {"warnings": ["센서 정보를 가져올 수 없습니다"]}

        battery = status.get("battery_level")
        temp = status.get("battery_temp")

        if battery is not None and battery <= self.battery_threshold:
            msg = f"스마트폰 배터리 부족: {battery}%"
            logger.warning(msg)
            warnings.append(msg)

        if temp is not None and temp >= self.temp_threshold:
            msg = f"스마트폰 과열! 배터리 온도: {temp}°C"
            logger.warning(msg)
            warnings.append(msg)

        return {
            "warnings": warnings,
            "battery_level": battery,
            "battery_temp": temp,
        }

    def get_status_string(self) -> str:
        """모니터링 화면에 표시할 상태 문자열을 반환한다."""
        status = self.check_status()
        battery = status.get("battery_level", "N/A")
        temp = status.get("battery_temp", "N/A")

        battery_str = f"{battery}%" if battery != "N/A" else "N/A"
        temp_str = f"{temp}°C" if temp != "N/A" else "N/A"

        return f"Battery: {battery_str}  |  Temp: {temp_str}"
