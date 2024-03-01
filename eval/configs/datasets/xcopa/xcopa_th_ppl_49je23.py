from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AnsEvaluator
from opencompass.datasets import XCOPASDataset

xcopa_reader_cfg = dict(
    input_columns=['question', 'premise', 'choice1', 'choice2'],
    output_column='answer')


system_prompt = '''โปรดอ่านหลักฐานที่ให้ไว้และตอบคำถาม\n'''

exp_1 = '''สถานที่: หมาเห่า
คำถาม: ผลกระทบคืออะไร?'''
ans_1 = '''คำตอบ: มีคนเคาะประตู\n'''

exp_2 = '''สถานที่: ผู้หญิงติดต่อนายหน้าอสังหาริมทรัพย์
คำถาม: ผลกระทบคืออะไร?'''
ans_2 = '''คำตอบ: ผู้หญิงวางแผนที่จะซื้อคอนโด\n'''

exp_3 = '''สถานที่: ผู้หญิงขอให้ผู้ชายออกไป
คำถาม: ผลกระทบคืออะไร?'''
ans_3 = '''คำตอบ: เขาสบประมาทผู้หญิง\n'''

prompt_input = '''สถานที่: {premise}
คำถาม: อะไรคือ {question}?'''


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
                    dict(role='BOT', prompt='คำตอบ: {{choice{ans}}}'.format(ans=int(ans)+1)),
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
        abbr='xcopa-th',
        path='./data/xcopa/th_test.jsonl',
        reader_cfg=xcopa_reader_cfg,
        infer_cfg=xcopa_infer_cfg,
        eval_cfg=xcopa_eval_cfg)
]