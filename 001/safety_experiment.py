import gc
import json
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
from experiment_config import ExperimentConfig
from model_connector import ModelConnector
from tqdm import tqdm


class SafetyExperiment:
    """安全性介入実験を実行するクラス"""

    def __init__(self, config=None):
        self.config = config or ExperimentConfig()
        self.connector = ModelConnector()

        # 実験設計のためのランダム化
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)

        # モデルの処置群・対照群への割り当て
        self.treatment_models, self.control_models = self._randomize_models()

        # 結果保存用のデータフレーム
        self.results_df = pd.DataFrame()

    def _randomize_models(self):
        """モデルをランダムに処置群と対照群に割り当て"""
        models = self.config.models.copy()
        random.shuffle(models)
        mid = len(models) // 2
        return models[:mid], models[mid:]

    def run_experiment(self, parallel=True, max_workers=5):
        """実験を実行"""
        print("実験を開始します...")

        # 実験条件のログ
        treatment_model_names = [m["name"] for m in self.treatment_models]
        control_model_names = [m["name"] for m in self.control_models]
        print(f"処置群モデル: {treatment_model_names}")
        print(f"対照群モデル: {control_model_names}")

        results = []

        # 全実験の総数
        total_experiments = len(self.treatment_models + self.control_models) * len(
            self.config.prompts
        )

        # プログレスバーの設定
        pbar = tqdm(total=total_experiments, desc="実験進捗")

        try:
            # 処置群のモデル
            for model in self.treatment_models:
                for prompt in self.config.prompts:
                    result = self._run_single_experiment(
                        model,
                        prompt,
                        f"{self.config.base_system_prompt} {self.config.treatment_instruction}",
                        "処置群",
                    )
                    results.append(result)
                    pbar.update(1)
                    gc.collect()

            # 対照群のモデル
            for model in self.control_models:
                for prompt in self.config.prompts:
                    result = self._run_single_experiment(
                        model, prompt, self.config.base_system_prompt, "対照群"
                    )
                    results.append(result)
                    pbar.update(1)
                    gc.collect()

            pbar.close()

            # 結果をデータフレームに変換
            self.results_df = pd.DataFrame(results)

        finally:
            # 実験終了後にメモリを解放
            self._clear_experiment()

        self._save_results()

        print(self.results_df.head())

    def _clear_experiment(self):
        """実験関連のメモリを解放"""
        self.treatment_models = None
        self.control_models = None
        gc.collect()

    def _run_single_experiment(self, model, prompt, system_prompt, group):
        """単一の実験を実行"""
        self.connector.treatment_instruction = self.config.treatment_instruction

        # モデルに応答を生成させる
        response = self.connector.generate_response(model, system_prompt, prompt)

        # 結果を辞書形式で返す
        result = {
            "model": model["name"],
            "company": model["company"],
            "model_size": model["size"],
            "provider": model["provider"],
            "group": group,
            "prompt": prompt,
            "system_prompt": system_prompt,
            "response": response,
        }

        return result

    def _save_results(self):
        """実験結果を保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        results_path = os.path.join(
            self.config.output_dir, f"experiment_results_{timestamp}.csv"
        )
        self.results_df.to_csv(results_path, index=False)
