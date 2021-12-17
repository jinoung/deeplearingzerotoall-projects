from pathlib import Path
import platform
import shutil
import os
import zipfile

def process_submit():
    plat_system = platform.system()
    # start at project directory
    project_path = Path().cwd().absolute()
    file_name = str(list(project_path.glob("*-project*.ipynb"))[0].relative_to(project_path))
    submit_file = project_path / file_name
    submit_rubric_file = project_path / "check_util" / f"{file_name.split('-')[1]}_submission.tsv"
    output_path = project_path / "submit"
    
    if not output_path.exists():
        output_path.mkdir()
    print(f"[ Self-Check ] 시스템: {plat_system}")
    
    # checking all pass or not
    x = submit_rubric_file.read_text(encoding="utf-8").splitlines()[1:]
    # remove blank first & end point
    x = [line.strip() for line in x]
    check_func = lambda x: True if x == "Pass" else False
    check_list = []
    for i, line in enumerate(x, 1):
        temp = line.split("\t")
        if check_func(temp[-1]):
            check_list.append(1)
        else:
            check_list.append(0)
            print("""[ Self-Check ] 
            [평가기준-{}] 통과하지 못했습니다. 다음 항목을 참고하세요!
            항목: '{}', 
            기준: '{}', 
            세부기준: '{}'""".format(i, *temp[:-1]))
    if sum(check_list) == len(x):
        # making submission files
        shutil.copy(str(submit_rubric_file), str(output_path))
        sub_string = f"jupyter nbconvert {str(submit_file)} --to html --output {file_name.split('-')[1]}_submission --output-dir={output_path}"
        os.system(sub_string)
        print(f"[ Self-Check ] Submit 파일 생성완료! 위치: '{(output_path).relative_to(project_path)}'")

        # zip files
        files = list(output_path.glob("*submission.*"))
        with zipfile.ZipFile("submit.zip", "w") as zip_handle:
            for f in files:
                zip_handle.write(str(f.relative_to(project_path)))
        print("[ Self-Check ] submit.zip 생성 완료!")
        print("[ Self-Check ] 모든 평가기준을 통과했습니다. 압축파일을 제출해주세요!")
    else:
        print("[ Self-Check ] 일부 평가기준을 통과하지 못했습니다. 제출 파일이 생성되지 않습니다. 다시 시도해보세요!")