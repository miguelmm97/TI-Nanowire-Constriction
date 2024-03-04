import pathlib
import tomli

path = pathlib.Path(__file__).parent / "variables.toml"
with path.open(mode="rb") as fp:
    variables = tomli.load(fp)

