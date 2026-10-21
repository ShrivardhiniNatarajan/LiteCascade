"""Write the SYNTHETIC fixture (N-BaIoT / CICIoT2023 / IoT-23 native layouts) to data/fixture/.

python scripts/make_fixture.py [--out data/fixture]
"""

import argparse

from litecascade.data.synthetic import DEFAULT_FIXTURE_DIR, write_fixture


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_FIXTURE_DIR))
    out = write_fixture(ap.parse_args().out)
    for p in sorted(out.rglob("*")):
        if p.is_file():
            print(p)


if __name__ == "__main__":
    main()
