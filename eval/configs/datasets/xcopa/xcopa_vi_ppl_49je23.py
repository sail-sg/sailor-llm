from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AnsEvaluator
from opencompass.datasets import XCOPASDataset

xcopa_reader_cfg = dict(
    input_columns=['question', 'premise', 'choice1', 'choice2'],
    output_column='answer')


system_prompt = '''Hãy đọc tiền đề đã cho và trả lời câu hỏi.\n'''

exp_1 = '''Tiền đề: Người đàn ông uống rất nhiều trong bữa tiệc.
Câu hỏi: Hiệu quả là gì?'''
ans_1 = '''Trả lời: Anh ấy bị đau đầu vào ngày hôm sau.\n'''

exp_2 = '''Tiền đề: Người chơi đã bắt được bóng.
Câu hỏi: Nguyên nhân là gì?'''
ans_2 = '''Trả lời: Đồng đội của cô ấy đã ném cho cô.\n'''

exp_3 = '''Tiền đề: Thư ký nói người gọi đợi.
Câu hỏi: Hiệu quả là gì?'''
ans_3 = '''Trả lời: Người gọi chờ.\n'''

prompt_input = '''Tiền đề: {premise}
Câu hỏi: cái gì là {question}?'''


xcopa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            ans: dict(
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
                    dict(role='BOT', prompt='Trả lời: {{choice{ans}}}'.format(ans=int(ans)+1)),
                ])
            for ans in ['0', '1']
        }),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer, generation_kwargs=dict(do_sample=False)))


xcopa_eval_cfg = dict(
    evaluator=dict(type=AnsEvaluator),
    pred_role="BOT")

xcopa_datasets = [
    dict(
        type=XCOPASDataset,
        abbr='xcopa-vi',
        path='./data/xcopa/vi_test.jsonl',
        reader_cfg=xcopa_reader_cfg,
        infer_cfg=xcopa_infer_cfg,
        eval_cfg=xcopa_eval_cfg)
]