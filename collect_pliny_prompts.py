from __future__ import annotations
from pathlib import Path
import pydantic
from typing import List
class Prompt(pydantic.BaseModel):
    header: str
    file: str
    prompt: str
class Prompts(pydantic.BaseModel):
    prompts: List[Prompt]


pliny_path = Path(__file__).parent / "dependencies" / "L1B3RT4S-git"
def get_pliny_prompts(pliny_path: Path = pliny_path) -> Prompts:
    if not pliny_path.exists():
        raise FileNotFoundError(f"Pliny path {pliny_path} does not exist")
    prompts = []
    for prompt_file in pliny_path.glob("**/*.mkd"):
        lines = [(i, line) for i, line in enumerate(prompt_file.read_text().split("\n"))]
        header_lines = [(i, line) for i, line in lines if line.startswith("# ")]
        for (i, header), (j, _) in zip(header_lines, header_lines[1:]):
            assert isinstance(i, int) and isinstance(j, int) and i < j
            assert isinstance(header, str) and header.startswith("# ")
            header_text = header[len("# "):].strip()
            prompt_text = "\n".join([z for _, z in lines[i+1:j]])
            prompts.append(Prompt(header=header_text, file=prompt_file.name, prompt=prompt_text))
            
    return Prompts(prompts=prompts)

def main():
    pliny_prompts = get_pliny_prompts(pliny_path)
    n_files = len(set((z.resolve().as_posix() for z in pliny_path.glob("**/*.mkd"))))
    print(pliny_prompts.model_dump_json(indent=4))
    print("="*100)
    print(f"Found {len(pliny_prompts.prompts)} prompts from {n_files} files")
    print("Example")
    print(pliny_prompts.prompts[0].model_dump_json(indent=4))
    print("="*100)
    print(pliny_prompts.prompts[0].prompt)

if __name__ == "__main__":
    main()