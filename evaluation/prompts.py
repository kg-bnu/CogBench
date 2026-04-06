# 用于答案生成的prompt
prompt_generate_answerr_only_title = """
Please generate an answer for the following math question, including a brief mathematical analysis:
Question: {title}
Please provide:
1. A brief problem-solving approach
2. Detailed calculation steps
3. The final answer
Please reply in brief academic Chinese.
"""
prompt_generate_answerr_title_grade = """
Please use the following math problem and grade information to generate an answer with a brief math analysis:
Question: {title}
Grade: {target_grade}
You must based on the knowledge level of students in this grade to answer question, please provide:
1. A suitable problem-solving approach for this grade
2. Detailed calculation steps
3. The final answer
Please reply in brief academic Chinese.
"""
prompt_generate_answerr_title_knowledge_text = """
please generate an answer with a brief mathematical analysis based on the following math problem and related knowledge points:
Question: {title}

Related Knowledge Points:
{knowledge_text}

You must based on the provided Related Knowledge Points to answer question:
1. A problem-solving approach based on the relevant knowledge points
2. Brief calculation steps
3. Final answer

Please reply in brief academic Chinese, don't exceed 500 words.
"""
eval_answer_prompt = """
As a professional teacher, you need to judge whether the student's answer is correct.
Student's answer: {response_answer}
Standard answer: {full_answer}
Brief answer: {brief_answer}
You must output your judgment result in one of the following two forms: True/False"""
