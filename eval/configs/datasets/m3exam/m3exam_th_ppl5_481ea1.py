from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer 
from opencompass.datasets import M3EXAMTHDataset, M3examThEvaluator


m3exam_reader_cfg = dict(
    input_columns=['question_text', 'options_0', 'options_1', 'options_2', 'options_3', 'options_4'],
    output_column='answer')

system_prompt = '''โปรดปฏิบัติตามตัวอย่างที่ให้มาและตอบคำถาม\n'''


exp_1 = '''คำถาม: ข้อใดเป็นหน้าที่ของพลเมืองไทยทุกคน'''
ans_1 = '''คำตอบ: เคารพกฎหมาย\n'''


exp_2 = '''คำถาม: บุคคลในข้อใดจัดเป็นอภัพพบุคคล ประเภทบุคคลที่ห้ามอุปสมบทแบบเด็ดขาด'''
ans_2 = '''คำตอบ: คนมีเพศบกพร่อง\n'''


exp_3 = '''คำถาม: ซื้อน้ำผลไม้ 9 กล่อง ราคากล่องละ 60.50 บาท จะต้องจ่ายเงินทั้งหมดกี่บาท'''
ans_3 = '''คำตอบ: 544.50\n'''


prompt_input = '''คำถาม: {question_text}'''
prompt_output = '''คำตอบ:'''


m3exam_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            str(ans+1): dict(
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
                    dict(role='BOT', prompt=f'คำตอบ: {{options_{ans}}}'),
                ])
            for ans in range(5)
        }),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer, generation_kwargs=dict(do_sample=False)))


m3exam_eval_cfg = dict(
    evaluator=dict(type=M3examThEvaluator),
    pred_role="BOT")

m3exam_datasets = [
    dict(
        type=M3EXAMTHDataset,
        abbr='m3exam-th-ppl-5',
        path='./data/m3exam/thai-questions-test-5.json',
        reader_cfg=m3exam_reader_cfg,
        infer_cfg=m3exam_infer_cfg,
        eval_cfg=m3exam_eval_cfg)
]