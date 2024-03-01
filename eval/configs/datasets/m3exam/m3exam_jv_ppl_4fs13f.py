from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.datasets import M3EXAMJvDataset, M3examEmEvaluator


m3exam_reader_cfg = dict(
    input_columns=['question_text', 'options_0', 'options_1', 'options_2', 'options_3'],
    output_column='answer_text')

system_prompt = '''Mangga tindakake conto diwenehi lan wangsulana pitakonan.\n'''


exp_1 = '''Pertanyaan: Pak Untung iku sabendinane lunga menyang sanggar saperlu mimpin lan ngatur lumakune crita drama\n\nMiturut wacan ing inggil, pendamelan (penggawean) pak Untung dados'''
ans_1 = '''Jawaban: Sutradara\n'''


exp_2 = '''Pertanyaan: Godhong Gedhang Garing Diarani'''
ans_2 = '''Jawaban: Klaras\n'''


exp_3 = '''Pertanyaan: Pandu iku yen lagi main drama mesti dadi wong sing becik, seneng mbiyantu lan sabar.\n\nSaking pranyatan ing inggil, Pandu niku minangka paraga'''
ans_3 = '''Jawaban: Protagonis\n'''


prompt_input = '''Pertanyaan: {question_text}'''
prompt_output = '''Jawaban:'''


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
                    dict(role='BOT', prompt=f'Jawaban: {{options_{ans}}}'),
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
        type=M3EXAMJvDataset,
        abbr='m3exam-id-ppl',
        path='./data/m3exam/javanese-questions-test.json',
        reader_cfg=m3exam_reader_cfg,
        infer_cfg=m3exam_infer_cfg,
        eval_cfg=m3exam_eval_cfg)
]