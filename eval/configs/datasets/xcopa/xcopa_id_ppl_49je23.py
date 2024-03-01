from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AnsEvaluator
from opencompass.datasets import XCOPASDataset

xcopa_reader_cfg = dict(
    input_columns=['question', 'premise', 'choice1', 'choice2'],
    output_column='answer')


system_prompt = '''Silakan baca premis yang diberikan dan jawab pertanyaannya.\n'''

exp_1 = '''Premis: Nyala api pada lilin itu sudah padam.
Pertanyaan: Apa penyebabnya?'''
ans_1 = '''Jawaban: Saya telah tiup sumbu lilin itu.\n'''

exp_2 = '''Premis: Hakim memukul-mukulkan palunya.
Pertanyaan: Apa efeknya?'''
ans_2 = '''Jawaban: Ruang sidang menjadi penuh keributan.\n'''

exp_3 = '''Premis: Langganan saya ke majalah itu telah kedaluwarsa.
Pertanyaan: Apa efeknya?'''
ans_3 = '''Jawaban: Saya berhenti menerima terbitannya yang baru.\n'''

prompt_input = '''Premis: {premise}
Pertanyaan: Apakah yang {question}?'''


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
                    dict(role='BOT', prompt='Jawaban: {{choice{ans}}}'.format(ans=int(ans)+1)),
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
        abbr='xcopa-id',
        path='./data/xcopa/id_test.jsonl',
        reader_cfg=xcopa_reader_cfg,
        infer_cfg=xcopa_infer_cfg,
        eval_cfg=xcopa_eval_cfg)
]