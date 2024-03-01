from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer 
from opencompass.datasets import M3EXAMVIDataset, M3examEmEvaluator


m3exam_reader_cfg = dict(
    input_columns=['question_text',  'options_0', 'options_1', 'options_2', 'options_3'],
    output_column='answer_text')

system_prompt = '''Hãy làm theo các ví dụ đã cho và trả lời câu hỏi.\n'''


exp_1 = '''Câu hỏi: Trong các câu sau đây, câu nào không chứa thành phần tình thái?'''
ans_1 = '''Trả lời: Trời ơi, chỉ còn năm phút.\n'''


exp_2 = '''Câu hỏi: Hỗn số $5\\frac{3}{5}$ được viết dưới dạng phân số là:'''
ans_2 = '''Trả lời: $28/5$\n'''


exp_3 = '''Câu hỏi: Xét về cấu tạo ngữ pháp, câu “Tác phẩm vừa là kết tinh của tâm hồn người sáng tác, vừa là sợi dây truyền cho mọi người sự sống mà nghệ sĩ mang trong lòng\" (Nguyễn Đình Thi Tiếng nói của văn nghệ) thuộc kiểu câu nào?'''
ans_3 = '''Trả lời: Câu đơn.\n'''


prompt_input = '''Câu hỏi: {question_text}'''
prompt_output = '''Trả lời:'''


m3exam_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            chr(65+ans): dict(
                begin=[
                    dict(role='SYSTEM', fallback_role='HUMAN', prompt=system_prompt),
                ],
                round=[
                    dict(role='HUMAN', prompt=exp_1),
                    dict(role='BOT', prompt=ans_1),
                    dict(role='HUMAN', prompt=exp_2),
                    dict(role='BOT', prompt=ans_2),
                    dict(role='HUMAN', prompt=exp_3),
                    dict(role='BOT', prompt=ans_3),
                    dict(role='HUMAN', prompt=prompt_input),
                    dict(role='BOT', prompt=f'Trả lời: {{options_{ans}}}'),
                ])
            for ans in range(4)
        }),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer, generation_kwargs=dict(do_sample=False)))


m3exam_eval_cfg = dict(
    evaluator=dict(type=M3examEmEvaluator),
    pred_role="BOT")

m3exam_datasets = [
    dict(
        type=M3EXAMVIDataset,
        abbr='m3exam-vi-ppl-4',
        path='./data/m3exam/vietnamese-questions-test-4.json',
        reader_cfg=m3exam_reader_cfg,
        infer_cfg=m3exam_infer_cfg,
        eval_cfg=m3exam_eval_cfg)
]