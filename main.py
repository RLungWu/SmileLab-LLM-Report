import os
import json
import ollama
from tqdm import tqdm
import logging
from datetime import datetime

# 設定日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_dataset(file_path):
    """載入JSONL格式的資料集"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"資料集檔案不存在: {file_path}")
    
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    logger.warning(f"第{line_num}行JSON解析錯誤: {e}")
                    continue
        logger.info(f"成功載入 {len(data)} 筆資料")
        return data
    except Exception as e:
        logger.error(f"載入資料集時發生錯誤: {e}")
        raise

def create_prompt(question, options, prompt_template=None):
    """創建提示詞"""
    if prompt_template is None:
        prompt_template = (
            "Please answer the following multiple choice question by selecting A, B, C, D, or E. "
            "Respond with only the letter of your choice.\n\n"
            "Question: {question}\n"
            "Options: {options}\n"
            "Answer:"
        )
    
    return prompt_template.format(question=question, options=options)

def query_model(model_name, prompt, max_retries=3):
    """查詢模型並處理重試邏輯"""
    for attempt in range(max_retries):
        try:
            response = ollama.chat(model=model_name, messages=[
                {"role": "user", "content": prompt}
            ])
            return response['message']['content'].strip()
        except Exception as e:
            logger.warning(f"模型查詢失敗 (嘗試 {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                logger.error(f"模型查詢最終失敗: {e}")
                return "ERROR"
    return "ERROR"

def extract_answer(response_text):
    """從模型回應中提取答案字母"""
    response_text = response_text.upper().strip()
    
    # 尋找A-E字母
    for char in response_text:
        if char in ['A', 'B', 'C', 'D', 'E']:
            return char
    
    # 如果找不到，返回原始回應
    return response_text

def evaluate_results(results):
    """評估結果並計算準確率"""
    total = len(results)
    correct = 0
    errors = 0
    
    for idx, result in results.items():
        if result['model_response'] == 'ERROR':
            errors += 1
        elif result['extracted_answer'] == result['correct_answer']:
            correct += 1
    
    accuracy = (correct / (total - errors)) * 100 if (total - errors) > 0 else 0
    
    return {
        'total_questions': total,
        'correct_answers': correct,
        'errors': errors,
        'accuracy': accuracy
    }

def save_results(results, model_name, output_dir="results"):
    """保存結果到檔案"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"usmle_results_{model_name}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    try:
        with open(filepath, "w", encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        logger.info(f"結果已保存至: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"保存結果失敗: {e}")
        raise

def main():
    # 配置參數
    CONFIG = {
        "dataset_path": "./data/USMLE.jsonl",
        "model_name": "gemma2",
        "output_dir": "results",
        "max_questions": None,  # None表示處理所有問題，或設定數字限制處理數量
    }
    
    try:
        # 載入資料集
        logger.info("開始載入資料集...")
        data_mle = load_dataset(CONFIG["dataset_path"])
        
        if CONFIG["max_questions"]:
            data_mle = data_mle[:CONFIG["max_questions"]]
            logger.info(f"限制處理前 {CONFIG['max_questions']} 個問題")
        
        results = {}
        
        # 處理每個問題
        logger.info(f"開始使用模型 {CONFIG['model_name']} 處理問題...")
        
        for idx, item in enumerate(tqdm(data_mle, desc="處理問題")):
            question = item['question']
            options = item['options']
            correct_answer = item['answer_idx']  # 正確答案
            
            # 創建提示詞
            prompt = create_prompt(question, options)
            
            # 查詢模型
            model_response = query_model(CONFIG["model_name"], prompt)
            
            # 提取答案
            extracted_answer = extract_answer(model_response)
            
            # 保存結果
            results[str(idx)] = {
                "question_id": idx,
                "model": CONFIG["model_name"],
                "question": question,
                "options": options,
                "correct_answer": correct_answer,
                "model_response": model_response,
                "extracted_answer": extracted_answer,
                "is_correct": extracted_answer == correct_answer
            }
            
            # 即時顯示進度（可選）
            if (idx + 1) % 10 == 0:
                logger.info(f"已處理 {idx + 1}/{len(data_mle)} 個問題")
        
        # 評估結果
        evaluation = evaluate_results(results)
        results["evaluation_summary"] = evaluation
        
        logger.info(f"評估完成:")
        logger.info(f"  總問題數: {evaluation['total_questions']}")
        logger.info(f"  正確答案: {evaluation['correct_answers']}")
        logger.info(f"  錯誤回應: {evaluation['errors']}")
        logger.info(f"  準確率: {evaluation['accuracy']:.2f}%")
        
        # 保存結果
        output_file = save_results(results, CONFIG["model_name"], CONFIG["output_dir"])
        logger.info("程式執行完成！")
        
        return output_file
        
    except Exception as e:
        logger.error(f"程式執行失敗: {e}")
        raise

if __name__ == "__main__":
    main()