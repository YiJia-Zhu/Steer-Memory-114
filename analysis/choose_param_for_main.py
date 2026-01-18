import pandas as pd

# 1. 读取原始 CSV
df = pd.read_csv("param.csv")

# 2. 确保 acc 是数值
df["acc"] = pd.to_numeric(df["acc"], errors="coerce")

# 3. 对每个 (models, dataset) 找 acc 最大的那一行
best_idx = (
    df
    .groupby(["models", "dataset"])["acc"]
    .idxmax()
)

best_df = df.loc[best_idx].reset_index(drop=True)

# 4. 保存（完整保留所有列）
best_df.to_csv("best_acc_full_metadata.csv", index=False)

print("✅ Saved best_acc_full_metadata.csv")



# import pandas as pd

# df = pd.read_csv("param.csv")

# df["tokens_used_mean"] = pd.to_numeric(df["tokens_used_mean"], errors="coerce")

# best_idx = (
#     df
#     .groupby(["models", "dataset"])["tokens_used_mean"]
#     .idxmin()
# )

# best_df = df.loc[best_idx].reset_index(drop=True)

# best_df.to_csv("min_token_full_metadata.csv", index=False)

# print("✅ Saved min_token_full_metadata.csv")
