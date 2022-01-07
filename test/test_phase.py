from tai_chi_tuna.config import PhaseConfig
from pathlib import Path
import os

TEST_DIR = Path(__file__).parent.parent

def test_phase():
    phase = PhaseConfig.load("./example_test")
    assert Path(phase.project).exists()==True
    phase['name'] = 'test'
    phase['description'] = 'test'
    del phase['description']
    phase.save()
    phase2 = PhaseConfig.load("./example_test")
    assert str(phase) == str(phase2)
    os.system("rm -rf ./example_test")

