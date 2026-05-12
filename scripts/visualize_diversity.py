import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 1. Data Loading
df = pd.read_parquet('data/processed/track_metadata.parquet')

try:
    # 2. Sorting
    if 'popularity' in df.columns:
        sort_col = 'popularity'
    elif 'count' in df.columns:
        sort_col = 'count'
    else:
        # Yedek plan: Eğer sütun adı farklıysa otomatik olarak bulmak için
        sort_col = df.columns[1] 

    df_sorted = df.sort_values(sort_col, ascending=False).reset_index(drop=True)
    y_values = df_sorted[sort_col].values
    x_values = np.arange(1, len(y_values) + 1)

    # 3. Canvas Preparation
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 6))

    # 4. Plotting Distribution
    ax.fill_between(x_values, y_values, color='#282828', alpha=0.8, label='Entire Catalog (Long Tail)')

    # 5. Shading Short-Head
    head_limit = int(len(x_values) * 0.2)
    ax.fill_between(x_values[:head_limit], y_values[:head_limit], color='#1DB954', alpha=0.4, label='Short Head (Collab Only)')

    # 6. Annotation
    y_text_pos = y_values[0] * 0.2 if y_values[0] > 0 else 20000
    ax.annotate('Hybrid Model Discovery Area (Diversity)', 
                xy=(len(x_values)*0.7, y_values[int(len(y_values)*0.1)]), 
                xytext=(len(x_values)*0.6, y_text_pos),
                arrowprops=dict(facecolor='#1DB954', shrink=0.05), 
                fontsize=12, color='#1DB954')

    # 7. Axis Labels
    ax.set_title('Popularity Distribution and Model Discovery Capacity', fontsize=16, pad=20, color='#f5f5f5')
    ax.set_xlabel('Song Popularity Ranking (Rank)', fontsize=12)
    ax.set_ylabel('Interaction / Play Count', fontsize=12)
    ax.legend(loc='upper right')

    # 8. Save & Show
    plt.tight_layout()
    plt.savefig('poster_long_tail_real.png', dpi=300)
    print("✅ Gerçek verilerinizle grafik 'poster_long_tail_real.png' olarak kaydedildi!")
    plt.show()

except Exception as e:
    print(f"Bir hata oluştu: {e}")
    print(f"Mevcut sütunlar (Columns): {df.columns.tolist()}")