from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_dashboard():
    root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(str(root / "dashboard.py")).run(timeout=30)
    assert not app.exception
    if (root / "reports/result.json").exists():
        assert len(app.dataframe) == 2
        app.selectbox[0].select("dev").run(timeout=30)
        assert not app.exception
