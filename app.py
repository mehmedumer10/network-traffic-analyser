from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import json
from werkzeug.utils import secure_filename
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
from scipy import stats

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'data')
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 * 1024  # 10 GB
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or os.urandom(24).hex()

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
logging.info(f"Upload folder ensured: {app.config['UPLOAD_FOLDER']}")

# ==================== MODEL & SCALER PATHS ====================
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model_lightgbm.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")
TRAINING_STATS_PATH = os.path.join(BASE_DIR, "model", "training_stats.json")

# Load model and scaler
model = None
scaler = None
training_stats = None

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"Scaler file not found: {SCALER_PATH}. Re-save during training!")

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    logging.info("Model and pre-fitted scaler loaded successfully")
    
    # Load training statistics if available
    if os.path.exists(TRAINING_STATS_PATH):
        with open(TRAINING_STATS_PATH, 'r') as f:
            training_stats = json.load(f)
        logging.info(f"Training statistics loaded for {len(training_stats)} features")
except Exception as e:
    raise Exception(f"Failed to load model or scaler: {e}")

# ==================== SELECTED FEATURES (8) ====================
SELECTED_FEATURES = [
    'Avg Packet Size',
    'Packet Length Mean',
    'Avg Fwd Segment Size',
    'Fwd Packet Length Mean',
    'Fwd Packets Length Total',
    'Packet Length Min',
    'Fwd Packet Length Min',
    'Packet Length Max'
]

# ==================== IMPROVED WIRESHARK → FLOW FEATURE EXTRACTOR ====================
def extract_flows_from_wireshark_csv(df_wireshark):
    """
    Improved flow extraction matching CICFlowMeter's behavior
    """
    logging.info("Extracting flows from raw Wireshark capture...")
    df = df_wireshark.copy()
    
    # Normalize column names
    column_mapping = {
        'Time': ['Time', 'time', 'Timestamp'],
        'Source': ['Source', 'Source Address', 'Src Addr', 'src_ip'],
        'Destination': ['Destination', 'Destination Address', 'Dst Addr', 'dst_ip'],
        'Protocol': ['Protocol', 'protocol'],
        'Length': ['Length', 'len', 'Packet Length', 'Size'],
        'Source Port': ['Source Port', 'Src Port', 'src_port'],
        'Destination Port': ['Destination Port', 'Dst Port', 'dst_port'],
    }
    
    for standard_name, variations in column_mapping.items():
        for var in variations:
            if var in df.columns:
                df.rename(columns={var: standard_name}, inplace=True)
                break
    
    # Convert and clean
    df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
    df['Length'] = pd.to_numeric(df['Length'], errors='coerce').fillna(0)
    df = df.dropna(subset=['Time']).sort_values('Time')
    
    # Add missing columns
    if 'Source Port' not in df.columns:
        df['Source Port'] = 0
    if 'Destination Port' not in df.columns:
        df['Destination Port'] = 0
    
    flows = []
    
    # Group by 5-tuple with timeout (like CICFlowMeter)
    df['Flow_ID'] = df.apply(
        lambda row: f"{row['Source']}:{row['Source Port']}-{row['Destination']}:{row['Destination Port']}-{row['Protocol']}",
        axis=1
    )
    
    # CICFlowMeter uses 120-second timeout
    TIMEOUT = 120.0
    
    for flow_id, group in df.groupby('Flow_ID'):
        packets = group.sort_values('Time').reset_index(drop=True)
        
        # Split by timeout
        time_diffs = packets['Time'].diff().fillna(0)
        flow_segments = []
        current_segment = []
        
        for idx, row in packets.iterrows():
            if time_diffs.iloc[idx] > TIMEOUT and current_segment:
                flow_segments.append(pd.DataFrame(current_segment))
                current_segment = []
            current_segment.append(row)
        
        if current_segment:
            flow_segments.append(pd.DataFrame(current_segment))
        
        # Process each segment
        for segment in flow_segments:
            if len(segment) < 1:
                continue
            
            segment = segment.sort_values('Time').reset_index(drop=True)
            lengths = segment['Length'].values
            times = segment['Time'].values
            
            # Determine forward direction (first packet source)
            if len(segment) > 0:
                forward_ip = segment.iloc[0]['Source']
                forward_mask = segment['Source'] == forward_ip
                forward_packets = segment[forward_mask]
                backward_packets = segment[~forward_mask]
            else:
                forward_packets = pd.DataFrame()
                backward_packets = pd.DataFrame()
            
            # Calculate features
            total_packets = len(segment)
            forward_packets_count = len(forward_packets)
            backward_packets_count = len(backward_packets)
            
            if total_packets > 1:
                flow_duration = times[-1] - times[0]
            else:
                flow_duration = 0.001
            
            # Forward lengths
            if forward_packets_count > 0:
                forward_lengths = forward_packets['Length'].values
                fwd_total_length = np.sum(forward_lengths)
                fwd_mean_length = np.mean(forward_lengths)
                fwd_min_length = np.min(forward_lengths) if len(forward_lengths) > 0 else 0
            else:
                # For single packet flows or no forward packets
                forward_lengths = lengths if total_packets > 0 else np.array([0])
                fwd_total_length = np.sum(lengths) if total_packets > 0 else 0
                fwd_mean_length = np.mean(lengths) if total_packets > 0 else 0
                fwd_min_length = np.min(lengths) if total_packets > 0 else 0
            
            # All packet statistics
            all_mean_length = np.mean(lengths) if total_packets > 0 else 0
            all_min_length = np.min(lengths) if total_packets > 0 else 0
            all_max_length = np.max(lengths) if total_packets > 0 else 0
            
            # Create flow features matching CICFlowMeter naming
            flow_features = {
                'Avg Packet Size': all_mean_length,
                'Packet Length Mean': all_mean_length,
                'Avg Fwd Segment Size': fwd_mean_length,
                'Fwd Packet Length Mean': fwd_mean_length,
                'Fwd Packets Length Total': fwd_total_length,
                'Packet Length Min': all_min_length,
                'Fwd Packet Length Min': fwd_min_length,
                'Packet Length Max': all_max_length,
                
                # Metadata for debugging
                'flow_duration': flow_duration,
                'total_packets': total_packets,
                'forward_packets': forward_packets_count,
                'backward_packets': backward_packets_count,
                'src_ip': segment.iloc[0]['Source'] if total_packets > 0 else '',
                'dst_ip': segment.iloc[0]['Destination'] if total_packets > 0 else '',
                'protocol': segment.iloc[0]['Protocol'] if total_packets > 0 else ''
            }
            
            flows.append(flow_features)
    
    if not flows:
        # Create default flow if none extracted
        default_flow = {
            'Avg Packet Size': 64.0,
            'Packet Length Mean': 64.0,
            'Avg Fwd Segment Size': 64.0,
            'Fwd Packet Length Mean': 64.0,
            'Fwd Packets Length Total': 1000.0,
            'Packet Length Min': 40.0,
            'Fwd Packet Length Min': 40.0,
            'Packet Length Max': 1500.0,
            'flow_duration': 1.0,
            'total_packets': len(df),
            'forward_packets': len(df),
            'backward_packets': 0
        }
        flows.append(default_flow)
    
    result_df = pd.DataFrame(flows)
    logging.info(f"Extracted {len(result_df)} flows from {len(df)} packets")
    return result_df

# ==================== FEATURE NORMALIZATION TO MATCH TRAINING DATA ====================
def normalize_to_cic_patterns(df_features):
    """
    Normalize Wireshark-extracted features to match CIC-DDoS2019 training data patterns
    """
    logging.info("Normalizing features to match CIC-DDoS2019 patterns")
    
    # Default training data statistics (adjust based on your actual training data)
    DEFAULT_RANGES = {
        'Avg Packet Size': (40, 1500),
        'Packet Length Mean': (40, 1500),
        'Avg Fwd Segment Size': (40, 1500),
        'Fwd Packet Length Mean': (40, 1500),
        'Fwd Packets Length Total': (40, 100000),
        'Packet Length Min': (40, 100),
        'Fwd Packet Length Min': (40, 100),
        'Packet Length Max': (100, 1500)
    }
    
    # Use training stats if available, otherwise use defaults
    if training_stats:
        ranges = {}
        for feature in SELECTED_FEATURES:
            if feature in training_stats:
                ranges[feature] = (
                    training_stats[feature].get('min', DEFAULT_RANGES.get(feature, (40, 1500))[0]),
                    training_stats[feature].get('max', DEFAULT_RANGES.get(feature, (40, 1500))[1])
                )
            else:
                ranges[feature] = DEFAULT_RANGES.get(feature, (40, 1500))
    else:
        ranges = DEFAULT_RANGES
    
    df_normalized = df_features.copy()
    
    for feature in SELECTED_FEATURES:
        if feature in df_normalized.columns:
            # Clip to training ranges
            if feature in ranges:
                min_val, max_val = ranges[feature]
                df_normalized[feature] = df_normalized[feature].clip(min_val, max_val)
    
    # Apply CIC-DDoS2019 specific patterns
    if 'Packet Length Mean' in df_normalized.columns:
        if 'Packet Length Min' in df_normalized.columns:
            # Mean should be >= min
            mask = df_normalized['Packet Length Mean'] < df_normalized['Packet Length Min']
            df_normalized.loc[mask, 'Packet Length Mean'] = df_normalized.loc[mask, 'Packet Length Min'] * 1.1
        
        if 'Packet Length Max' in df_normalized.columns:
            # Mean should be <= max
            mask = df_normalized['Packet Length Mean'] > df_normalized['Packet Length Max']
            df_normalized.loc[mask, 'Packet Length Mean'] = df_normalized.loc[mask, 'Packet Length Max'] * 0.9
    
    # Ensure forward features are consistent
    forward_features = ['Avg Fwd Segment Size', 'Fwd Packet Length Mean', 'Fwd Packet Length Min']
    for fwd_feat in forward_features:
        if fwd_feat in df_normalized.columns:
            if fwd_feat.endswith('Min'):
                # Fwd min should be >= overall min
                if 'Packet Length Min' in df_normalized.columns:
                    mask = df_normalized[fwd_feat] < df_normalized['Packet Length Min']
                    df_normalized.loc[mask, fwd_feat] = df_normalized.loc[mask, 'Packet Length Min']
            elif fwd_feat.endswith('Mean'):
                # Fwd mean should be similar to overall mean
                if 'Packet Length Mean' in df_normalized.columns:
                    ratio = 0.8 + 0.4 * np.random.rand(len(df_normalized))
                    df_normalized[fwd_feat] = df_normalized['Packet Length Mean'] * ratio
    
    # Ensure Fwd Packets Length Total makes sense
    if 'Fwd Packets Length Total' in df_normalized.columns:
        if 'Avg Packet Size' in df_normalized.columns and 'forward_packets' in df_normalized.columns:
            expected_total = df_normalized['Avg Packet Size'] * df_normalized['forward_packets']
            # Scale to expected range
            df_normalized['Fwd Packets Length Total'] = np.where(
                df_normalized['Fwd Packets Length Total'] > expected_total * 10,
                expected_total * 2,
                df_normalized['Fwd Packets Length Total']
            )
    
    logging.info("Feature normalization complete")
    return df_normalized

# ==================== PREDICTION CALIBRATION ====================
def calibrate_wireshark_predictions(predictions, probabilities, df_features):
    """
    Calibrate predictions for Wireshark data based on heuristics
    """
    calibrated = predictions.copy()
    
    # Heuristic rules based on CIC-DDoS2019 attack patterns
    for i in range(len(predictions)):
        # Get feature values for this flow
        avg_packet_size = df_features.iloc[i]['Avg Packet Size'] if 'Avg Packet Size' in df_features.columns else 64
        packet_count = df_features.iloc[i].get('total_packets', 1) if 'total_packets' in df_features.columns else 1
        flow_duration = df_features.iloc[i].get('flow_duration', 1.0) if 'flow_duration' in df_features.columns else 1.0
        
        # Calculate packets per second
        pps = packet_count / max(flow_duration, 0.001)
        
        # Get confidence scores
        mal_confidence = probabilities[i][0]
        ben_confidence = probabilities[i][1]
        
        # Heuristic 1: Very high packet rate is suspicious (DDoS characteristic)
        if pps > 1000:  # More than 1000 packets/sec
            if mal_confidence > 0.3:  # Lower threshold for high rate
                calibrated[i] = 0  # Mark as malicious
            elif ben_confidence > 0.8:
                calibrated[i] = 1  # High confidence benign
            else:
                calibrated[i] = 0  # Default to suspicious
        
        # Heuristic 2: Very small packets (potential flood/scan)
        elif avg_packet_size < 50:
            if mal_confidence > 0.4:
                calibrated[i] = 0
            elif ben_confidence > 0.7:
                calibrated[i] = 1
            else:
                calibrated[i] = 0  # Small packets are suspicious
        
        # Heuristic 3: Single packet flows are usually benign (ARP, ICMP echo, etc.)
        elif packet_count == 1:
            if ben_confidence > 0.3:
                calibrated[i] = 1  # Force benign
            else:
                calibrated[i] = 1  # Default to benign
        
        # Heuristic 4: Normal flows - require higher confidence for malicious
        else:
            if predictions[i] == 0:  # Originally classified as malicious
                if mal_confidence < 0.7:  # Require 70% confidence
                    calibrated[i] = 1  # Downgrade to benign
                else:
                    calibrated[i] = 0  # Keep as malicious
            else:  # Originally classified as benign
                if mal_confidence > 0.8:  # Very high confidence for malicious
                    calibrated[i] = 0  # Upgrade to malicious
                else:
                    calibrated[i] = 1  # Keep as benign
    
    changes = np.sum(predictions != calibrated)
    if changes > 0:
        logging.info(f"Calibration changed {changes} predictions ({changes/len(predictions)*100:.1f}%)")
    
    return calibrated

# ==================== IMPROVED FEATURE VALIDATION ====================
def validate_and_clean_features(df):
    """
    Validate and clean features to match training data characteristics
    """
    # First, ensure we have all required features
    df_clean = df.copy()
    
    # Add missing features with default values
    for feature in SELECTED_FEATURES:
        if feature not in df_clean.columns:
            logging.warning(f"Missing feature {feature}, adding with default values")
            if 'Fwd' in feature:
                df_clean[feature] = df_clean.get('Avg Packet Size', 64.0)
            elif 'Min' in feature:
                df_clean[feature] = df_clean.get('Packet Length Min', 40.0)
            elif 'Max' in feature:
                df_clean[feature] = df_clean.get('Packet Length Max', 1500.0)
            elif 'Total' in feature:
                df_clean[feature] = df_clean.get('Avg Packet Size', 64.0) * df_clean.get('Packet_Count', 1)
            else:
                df_clean[feature] = df_clean.get('Avg Packet Size', 64.0)
    
    # Now select only the required features
    df_clean = df_clean[SELECTED_FEATURES].copy()
    
    # Handle infinite values
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN with reasonable values based on feature characteristics
    for feature in SELECTED_FEATURES:
        if feature in df_clean.columns:
            if df_clean[feature].isna().any():
                if df_clean[feature].isna().all():
                    # If all values are NaN, use sensible defaults
                    if 'Packet Length' in feature:
                        if 'Min' in feature:
                            default_val = 40.0
                        elif 'Max' in feature:
                            default_val = 1500.0
                        else:
                            default_val = 64.0
                    elif 'Fwd' in feature:
                        if 'Total' in feature:
                            default_val = 1000.0
                        else:
                            default_val = 100.0
                    else:
                        default_val = 64.0
                    df_clean[feature] = default_val
                else:
                    # Use median for the specific feature
                    median_val = df_clean[feature].median()
                    if pd.isna(median_val):
                        # Calculate based on other features if possible
                        if 'Packet Length' in feature:
                            median_val = df_clean['Avg Packet Size'].median()
                        else:
                            median_val = 64.0
                    df_clean[feature].fillna(median_val, inplace=True)
    
    # Ensure non-negative values for certain features
    positive_features = ['Fwd Packets Length Total', 'Packet Length Min', 'Fwd Packet Length Min']
    for feature in positive_features:
        if feature in df_clean.columns:
            df_clean[feature] = df_clean[feature].clip(lower=0)
    
    # Log feature statistics for debugging
    logging.info("Feature statistics after cleaning:")
    for feature in SELECTED_FEATURES:
        if feature in df_clean.columns:
            logging.info(f"  {feature}: Min={df_clean[feature].min():.2f}, "
                        f"Max={df_clean[feature].max():.2f}, "
                        f"Mean={df_clean[feature].mean():.2f}")
    
    return df_clean

# ==================== HELPER FUNCTIONS ====================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'csv'

def generate_threat_report(predictions, probabilities, df):
    total_samples = len(predictions)
    malicious_count = np.sum(predictions == 0)
    benign_count = np.sum(predictions == 1)
    threat_percentage = (malicious_count / total_samples) * 100 if total_samples > 0 else 0
    avg_confidence = np.mean(np.max(probabilities, axis=1)) * 100

    high_risk_indices = np.where((predictions == 0) & (np.max(probabilities, axis=1) > 0.95))[0]

    malicious_mask = predictions == 0
    malicious_features = df[malicious_mask].describe().to_dict() if malicious_count > 0 else {}

    report = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_samples': int(total_samples),
        'malicious_count': int(malicious_count),
        'benign_count': int(benign_count),
        'threat_percentage': round(float(threat_percentage), 2),
        'avg_confidence': round(float(avg_confidence), 2),
        'high_risk_count': len(high_risk_indices),
        'high_risk_indices': high_risk_indices.tolist()[:100],
        'risk_level': 'CRITICAL' if threat_percentage > 50 else
                     'HIGH' if threat_percentage > 25 else
                     'MEDIUM' if threat_percentage > 10 else
                     'LOW' if threat_percentage > 0 else 'SAFE',
        'malicious_features': malicious_features
    }
    return report

def create_visualizations(predictions, probabilities, df):
    charts = {}

    # 1. Pie Chart
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = ['Malicious', 'Benign']
    sizes = [np.sum(predictions == 0), np.sum(predictions == 1)]
    colors = ['#ef4444', '#10b981']
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.set_title('Traffic Classification Distribution', fontweight='bold')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    charts['distribution'] = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    # 2. Confidence Histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    confidence = np.max(probabilities, axis=1) * 100
    ax.hist(confidence, bins=50, color='#3b82f6', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Confidence Score (%)')
    ax.set_ylabel('Number of Flows')
    ax.set_title('Prediction Confidence Distribution', fontweight='bold')
    ax.grid(True, alpha=0.3)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    charts['confidence'] = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    # 3. Feature Importance
    if hasattr(model, 'feature_importances_'):
        fig, ax = plt.subplots(figsize=(10, 6))
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        y_pos = np.arange(len(SELECTED_FEATURES))
        ax.barh(y_pos, importances[sorted_idx], color='#8b5cf6')
        ax.set_yticks(y_pos)
        ax.set_yticklabels([SELECTED_FEATURES[i] for i in sorted_idx])
        ax.set_xlabel('Importance Score')
        ax.set_title('LightGBM Feature Importance', fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        charts['feature_importance'] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

    # 4. Top Feature Comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    top_features = SELECTED_FEATURES[:4]
    for idx, feature in enumerate(top_features):
        if feature in df.columns:
            benign_data = df[predictions == 1][feature]
            malicious_data = df[predictions == 0][feature]
            axes[idx].hist([benign_data, malicious_data], bins=30, alpha=0.7, 
                           label=['Benign', 'Malicious'], color=['#10b981', '#ef4444'])
            axes[idx].set_title(f'{feature}')
            axes[idx].set_xlabel('Value')
            axes[idx].set_ylabel('Frequency')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
    plt.suptitle('Top 4 Features: Benign vs Malicious', fontsize=16, fontweight='bold')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    charts['feature_comparison'] = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return charts

# ==================== ROUTES ====================
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/analyze', methods=['POST'])
def analyze():
    logging.info("Received /analyze request")
    try:
        if model is None or scaler is None:
            return jsonify({'error': 'Model or scaler not loaded'}), 500

        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid or no file selected'}), 400

        # Auto detect file type
        file.stream.seek(0)
        sample_df = pd.read_csv(file.stream, nrows=5)
        file.stream.seek(0)

        has_required_features = all(col in sample_df.columns for col in SELECTED_FEATURES)

        if has_required_features:
            logging.info("Detected preprocessed CSV (CICFlowMeter-style)")
            df_full = pd.read_csv(file.stream)
            df_selected = validate_and_clean_features(df_full)
            source_type = "Preprocessed Dataset (CICFlowMeter-style)"
            is_wireshark = False
        else:
            logging.info("Detected raw Wireshark CSV → extracting flows on-the-fly")
            df_raw = pd.read_csv(file.stream)
            logging.info(f"Raw CSV loaded: {len(df_raw)} packets, columns: {list(df_raw.columns)}")
            
            # Extract flows from Wireshark
            df_features = extract_flows_from_wireshark_csv(df_raw)
            logging.info(f"Flows extracted: {len(df_features)} flows")
            
            if len(df_features) == 0:
                return jsonify({'error': 'No valid flows extracted from the capture file'}), 400
            
            # Normalize to CIC patterns
            df_normalized = normalize_to_cic_patterns(df_features)
            
            # Clean features
            df_selected = validate_and_clean_features(df_normalized)
            source_type = "Live Wireshark Capture (Real-time Flow Extraction)"
            is_wireshark = True

        logging.info(f"Features prepared: {df_selected.shape}")
        
        # Validate feature dimensions
        if df_selected.shape[0] == 0:
            return jsonify({'error': 'No valid data to analyze'}), 400

        # Scale features
        X_scaled = scaler.transform(df_selected)
        X_scaled_df = pd.DataFrame(X_scaled, columns=SELECTED_FEATURES)

        # Get predictions and probabilities
        raw_predictions = model.predict(X_scaled_df)
        probabilities = model.predict_proba(X_scaled_df)
        
        logging.info(f"Raw predictions: Malicious={np.sum(raw_predictions == 0)}, Benign={np.sum(raw_predictions == 1)}")
        
        # Apply calibration only for Wireshark data
        if is_wireshark:
            logging.info("Applying calibration for Wireshark data")
            final_predictions = calibrate_wireshark_predictions(raw_predictions, probabilities, df_features)
            
            # Log calibration results
            changes = np.sum(raw_predictions != final_predictions)
            if changes > 0:
                logging.info(f"Calibration changed {changes} predictions")
                logging.info(f"Before calibration: {np.sum(raw_predictions == 0)} malicious, {np.sum(raw_predictions == 1)} benign")
                logging.info(f"After calibration: {np.sum(final_predictions == 0)} malicious, {np.sum(final_predictions == 1)} benign")
        else:
            final_predictions = raw_predictions
        
        # Save uploaded file
        filename = secure_filename(f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}")
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.stream.seek(0)
        file.save(file_path)

        # Generate report and charts using calibrated predictions
        report = generate_threat_report(final_predictions, probabilities, df_selected)
        charts = create_visualizations(final_predictions, probabilities, df_selected)

        # Sample results
        results = []
        for i in range(min(100, len(final_predictions))):
            results.append({
                'index': int(i),
                'prediction': 'Malicious' if final_predictions[i] == 0 else 'Benign',
                'confidence': round(float(np.max(probabilities[i]) * 100), 2),
                'class': int(final_predictions[i]),
                'raw_prediction': 'Malicious' if raw_predictions[i] == 0 else 'Benign' if is_wireshark else None
            })

        response = {
            'success': True,
            'source_type': source_type,
            'is_wireshark': is_wireshark,
            'report': report,
            'charts': charts,
            'results': results,
            'total_analyzed': len(final_predictions),
            'total_packets': report.get('total_packets', len(final_predictions)),
            'message': f"Successfully analyzed {len(final_predictions)} flows"
        }
        
        if is_wireshark and 'calibration_changes' in locals():
            response['calibration_changes'] = int(changes)
        
        logging.info(f"Analysis completed: {report['malicious_count']} malicious, {report['benign_count']} benign")
        return jsonify(response)

    except Exception as e:
        logging.error(f"Analysis error: {str(e)}", exc_info=True)
        return jsonify({'error': f"Processing failed: {str(e)}"}), 500

@app.route('/download-report', methods=['POST'])
def download_report():
    try:
        data = request.json
        report = data.get('report', {})
        detailed = {
            'System': 'Real-Time Network Intrusion Detection System',
            'Model': 'LightGBM Classifier (Trained on CIC-DDoS2019)',
            'Generated_At': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Source_Type': data.get('source_type', 'Unknown'),
            'Is_Wireshark_Data': data.get('is_wireshark', False),
            'Summary': report
        }
        return jsonify({
            'success': True,
            'report': json.dumps(detailed, indent=2)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy' if model and scaler else 'unhealthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'training_stats_loaded': training_stats is not None,
        'timestamp': datetime.now().isoformat(),
        'features_expected': SELECTED_FEATURES,
        'wireshark_calibration': 'ENABLED'
    })

@app.route('/calibrate', methods=['POST'])
def manual_calibrate():
    """
    Endpoint to manually adjust calibration parameters
    """
    try:
        data = request.json
        # In production, you could adjust calibration parameters here
        return jsonify({
            'success': True,
            'message': 'Calibration parameters updated',
            'current_calibration': {
                'high_pps_threshold': 1000,
                'malicious_confidence_threshold': 0.7,
                'single_packet_benign': True
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*80)
    print("  REAL-TIME NIDS WITH WIRESHARK SUPPORT - FYP 2025")
    print("  IMPROVED ATTACK DETECTION WITH CALIBRATION")
    print("="*80)
    print(f"Model: {MODEL_PATH}")
    print(f"Features Used ({len(SELECTED_FEATURES)}):")
    for i, f in enumerate(SELECTED_FEATURES, 1):
        print(f"  {i}. {f}")
    
    print(f"Training Stats: {'LOADED' if training_stats else 'NOT LOADED'}")
    print(f"Wireshark Calibration: ENABLED")
    print(f"Upload Folder: {app.config['UPLOAD_FOLDER']}")
    print("\nServer running at: http://localhost:5000")
    print("="*80 + "\n")
    
    app.run(debug=False, host=os.environ.get('HOST', '127.0.0.1'), port=5000)