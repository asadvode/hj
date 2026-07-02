import importlib
import os
import unittest


class ApiFlowTests(unittest.TestCase):
    def test_generate_avatar_image_requires_configured_key(self):
        module = importlib.import_module("webui_orchestrator")
        portrait_path = module.ROOT / "portrait.jpg"
        if portrait_path.exists():
            portrait_path.unlink()

        module.NVIDIA_KEYS = {"flux_klein_key": ""}
        success, result = module.generate_avatar_image("demo prompt")

        self.assertFalse(success)
        self.assertIn("API key", result)
        self.assertFalse(portrait_path.exists())


if __name__ == "__main__":
    unittest.main()
