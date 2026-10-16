"""
Data provenance: how data/features/pitcher_games_enhanced.csv was built.

This script pulls Statcast pitch-by-pitch data via ``pybaseball``, keeps
first-inning pitches, and aggregates them per pitcher-game into the real
first-inning pitch count (``pitches_inning1_true``) plus per-game Statcast
summaries. It writes a local SQLite database and the committed CSV.

It is NOT needed to reproduce the evaluation (src/evaluate.py reads the
committed CSV directly). It requires a network connection and the extra
dependencies listed under "scrape only" in requirements.txt, and the raw
Statcast download is not included in this repository.
"""

import os
import pybaseball as pb
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import logging
from tqdm import tqdm
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

def get_db_connection():
    return create_engine("sqlite:///data/mlb_predictor.db")

def fetch_statcast_enhanced(start_date='2023-01-01', end_date='2025-12-31'):
    """Chunked Statcast + cache"""
    pb.cache.enable()
    all_data = []
    
    current = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    with tqdm(desc="Months", unit="month") as pbar:
        while current <= end:
            chunk_end = min(current + timedelta(days=90), end)
            try:
                chunk = pb.statcast(start_dt=current.strftime('%Y-%m-%d'), 
                                  end_dt=chunk_end.strftime('%Y-%m-%d'))
                if not chunk.empty:
                    all_data.append(chunk)
            except Exception as exc:
                logger.warning(f"chunk {current:%Y-%m-%d} failed: {exc}")
            current = chunk_end + timedelta(days=1)
            pbar.update(1)

    df = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    logger.info(f"{len(df):,} total pitches")
    return df

def engineer_true_pitch_counts(statcast_df):
    """TRUE inning 1 pitch totals + rich features"""
    if statcast_df.empty:
        return pd.DataFrame()
    
    # Filter inning 1
    inning1 = statcast_df[statcast_df['inning'] == 1].copy()
    
    # TRUE PITCH COUNT = total pitches per pitcher-game
    pitch_counts = inning1.groupby(['pitcher', 'game_pk']).size().reset_index(name='pitches_inning1_true')
    
    # Rich Statcast features
    agg_features = inning1.groupby(['pitcher', 'game_pk']).agg({
        'release_speed': ['mean', 'std'],
        'release_spin_rate': 'mean',
        'pitch_number': 'max',
        'at_bat_number': 'nunique',  # PAs faced
        'events': lambda x: (x == 'strikeout').sum(),  # K's
        'plate_x': 'std',  # Command consistency
        'release_pos_x': 'mean',  # Release point
        'pfx_x': 'mean',  # Horizontal movement
        'pfx_z': 'mean'   # Vertical movement
    }).round(2)
    
    # Flatten multi-index columns
    agg_features.columns = ['velo_avg', 'velo_std', 'spin_avg', 'pitch_max', 
                          'pas_faced', 'ks_inning1', 'command_std', 
                          'release_x', 'break_horz', 'break_vert']
    
    # Merge
    df_final = pitch_counts.merge(agg_features.reset_index(), on=['pitcher', 'game_pk'])
    
    logger.info(f"true pitch counts: {df_final['pitches_inning1_true'].mean():.1f} avg")
    logger.info(f"samples: {len(df_final):,}")
    return df_final

def save_enhanced_data(df, engine):
    """Enhanced schema - FIXED index error"""
    try:
        with engine.begin() as conn:
            # Drop old table if exists
            conn.execute(text("DROP TABLE IF EXISTS pitcher_games_enhanced"))
            
            # Save new data
            df.to_sql('pitcher_games_enhanced', conn, if_exists='replace', index=False)
            
            # Create indexes SAFELY (IF NOT EXISTS)
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_pitcher_game ON pitcher_games_enhanced(pitcher, game_pk)",
                "CREATE INDEX IF NOT EXISTS idx_pitcher ON pitcher_games_enhanced(pitcher)",
                "CREATE INDEX IF NOT EXISTS idx_game ON pitcher_games_enhanced(game_pk)"
            ]
            
            for idx_sql in indexes:
                conn.execute(text(idx_sql))
        
        logger.info("enhanced data + indexes saved")
    except Exception as e:
        logger.error(f"save failed: {e}")

def main():
    print("Building first-inning pitch counts + Statcast features from Statcast")

    engine = get_db_connection()
    data = fetch_statcast_enhanced()

    if data.empty:
        logger.error("no Statcast data")
        return

    enhanced_data = engineer_true_pitch_counts(data)
    save_enhanced_data(enhanced_data, engine)
    enhanced_data.to_csv('data/features/pitcher_games_enhanced.csv', index=False)

    print(f"pitcher-games: {len(enhanced_data):,}")
    print(f"true first-inning avg: {enhanced_data['pitches_inning1_true'].mean():.1f} pitches")
    print(f"range: {enhanced_data['pitches_inning1_true'].min()} - {enhanced_data['pitches_inning1_true'].max()}")

if __name__ == "__main__":
    main()
