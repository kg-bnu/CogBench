import json
import re


def extract_grades_from_solutions(solutions):
    """
    从solution数组中提取所有不同的年级标签，忽略大小写
    """
    grades = []
    for solution in solutions:
        match = re.match(r"(?i)grade\s*(\d+)", solution)
        if match:
            grade_num = match.group(1)
            grades.append(f"grade{grade_num}")
        else:
            grades.append("")
    return grades


def readCQAData(filePath: str):
    with open(filePath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def get_grade_num(grade_str):
    """从 'grade5' 或 'Grade3' 这样的字符串中提取数字"""
    if not grade_str:
        return None
    # 修改正则表达式，使其不区分大小写
    match = re.search(r"(?i)grade(\d+)", grade_str)
    if match:
        return int(match.group(1))
    return None


def parse_solution(solution_text):
    """从 'Grade3: ...' 或 'grade5: ...' 格式中分离年级和内容"""
    match = re.match(r"(?i)grade\s*(\d+)\s*[:：]?\s*(.*)", solution_text, re.DOTALL)
    if match:
        grade_str = match.group(1).strip()
        content = match.group(2).strip()
        return grade_str, content
    return None, None
