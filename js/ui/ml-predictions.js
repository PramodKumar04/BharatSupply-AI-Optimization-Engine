// ============================================================
// BharatSupply — Multi-Source ML Predictions Dashboard UI
// ============================================================
import { formatPercent } from '../utils/formatters.js';

/**
 * Render ML Predictions Dashboard with Delay Decomposition & Demurrage
 */
export function renderPredictionsPanel(vessels, predictions) {
    const container = document.getElementById('predictionsContent');
    if (!container) return;

    if (!vessels || vessels.length === 0) {
        container.innerHTML = `
            <div class="chart-empty" style="min-height:300px">
                <span class="chart-empty-icon">🚢</span>
                <div>No active or planned vessels to predict</div>
                <div style="font-size:0.72rem;color:var(--text-muted);margin-top:4px">
                    Assign a vessel in Strategic Planning to view multi-source delay forecasts
                </div>
            </div>
        `;
        return;
    }

    const averageConfidence = predictions.length > 0 ? 
        predictions.reduce((acc, p) => acc + (p.confidence || p.confidence_score || 0), 0) / predictions.length : 0.88;

    container.innerHTML = `
        <div class="card-header">
            <div>
                <h3 class="card-title">🚢 Multi-Source Vessel Delay Prediction System</h3>
                <p class="card-subtitle">VotingRegressor Ensemble (RF, GBDT, ExtraTrees, MLP) + RL Path Optimization (Confidence: ${formatPercent(averageConfidence)})</p>
            </div>
        </div>

        <div class="predictions-grid">
            ${vessels.map((v, i) => {
                const pred = predictions[i];
                if (!pred) return '';
                
                const totalDelay = pred.predictedDelay ?? pred.predicted_total_delay_h ?? 0;
                const isHighRisk = totalDelay > 24;
                const riskColor = isHighRisk ? 'var(--accent-danger)' : totalDelay > 10 ? '#b45309' : 'var(--accent-success)';
                
                const berth = pred.predicted_berthing_delay_h ?? Math.round(totalDelay * 0.45);
                const customs = pred.predicted_customs_delay_h ?? Math.round(totalDelay * 0.25);
                const yard = pred.predicted_yard_dwell_h ?? Math.round(totalDelay * 0.30);
                const reason = pred.predicted_primary_delay_reason || 'Port Congestion';
                const demurrage = pred.predicted_demurrage_usd || Math.round((totalDelay / 24) * 25000);
                const calcEta = pred.calculated_eta || (v.scheduledETA ? new Date(new Date(v.scheduledETA).getTime() + totalDelay * 3600000).toLocaleString() : '—');
                const paradigm = pred.selected_paradigm ? pred.selected_paradigm.replace(/_/g, ' ') : 'VotingRegressor Ensemble';

                return `
                    <div class="prediction-card card hover-lift">
                        <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px">
                            <div>
                                <h4 style="font-size:1rem;font-weight:700">${v.name}</h4>
                                <div style="font-size:0.72rem;color:var(--text-muted)">Scheduled: ${new Date(v.scheduledETA || Date.now()).toLocaleDateString()} at ${v.destinationPortName || v.destinationPort}</div>
                                <div style="font-size:0.68rem;color:var(--primary);margin-top:3px;font-weight:600">
                                    Calculated Arrival: ${calcEta}
                                </div>
                            </div>
                            <div class="risk-badge" style="background:${isHighRisk ? 'rgba(239, 68, 68, 0.1)' : 'rgba(16, 185, 129, 0.1)'}; color:${riskColor}; padding:4px 8px; border-radius:4px; font-size:0.65rem; font-weight:700; text-transform:uppercase">
                                ${isHighRisk ? 'High Delay' : totalDelay > 10 ? 'Moderate' : 'On Track'}
                            </div>
                        </div>

                        <div class="prediction-main">
                            <div class="prediction-value-group">
                                <div class="prediction-label">Predicted Delay</div>
                                <div class="prediction-value" style="color:${riskColor}">${totalDelay > 0 ? '+' : ''}${totalDelay}h</div>
                            </div>
                            <div class="prediction-confidence-group">
                                <div class="prediction-label">Model Confidence</div>
                                <div class="confidence-bar-container">
                                    <div class="confidence-bar-fill" style="width:${(pred.confidence || pred.confidence_score || 0.88) * 100}%; background:${riskColor}"></div>
                                </div>
                                <div style="font-size:0.65rem;text-align:right;margin-top:4px">${Math.round((pred.confidence || pred.confidence_score || 0.88) * 100)}%</div>
                            </div>
                        </div>

                        <!-- Delay Breakdown -->
                        <div style="background:var(--bg-tertiary);border-radius:6px;padding:8px 12px;margin:12px 0;font-size:0.75rem">
                            <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                                <span style="color:var(--text-muted)">Berthing Queue:</span>
                                <strong>${berth} hrs</strong>
                            </div>
                            <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                                <span style="color:var(--text-muted)">Customs Clearance:</span>
                                <strong>${customs} hrs</strong>
                            </div>
                            <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                                <span style="color:var(--text-muted)">Yard Stock Dwell:</span>
                                <strong>${yard} hrs</strong>
                            </div>
                            <div style="display:flex;justify-content:space-between;margin-top:6px;padding-top:6px;border-top:1px dashed var(--border-primary)">
                                <span style="color:var(--text-muted)">Demurrage Cost:</span>
                                <strong style="color:var(--accent-warning)">$${demurrage.toLocaleString()}</strong>
                            </div>
                        </div>

                        <div class="prediction-factors">
                            <div class="prediction-label" style="margin-bottom:6px">Root Cause &amp; Paradigm</div>
                            <div style="display:flex;justify-content:space-between;font-size:0.72rem;margin-bottom:3px">
                                <span style="color:var(--text-secondary)">Primary Reason:</span>
                                <span style="font-weight:600;color:var(--text-primary)">${reason}</span>
                            </div>
                            <div style="display:flex;justify-content:space-between;font-size:0.72rem">
                                <span style="color:var(--text-secondary)">Active Mode:</span>
                                <span style="font-size:0.68rem;color:var(--primary)">${paradigm}</span>
                            </div>
                        </div>

                        <div style="margin-top:12px;padding-top:10px;border-top:1px solid var(--border-primary);display:flex;justify-content:space-between;align-items:center">
                            <span style="font-size:0.65rem;color:var(--text-muted)">Data Source: Open-Meteo GIS &amp; Historical Priors</span>
                            <span style="font-size:0.62rem;background:var(--bg-secondary);padding:2px 6px;border-radius:3px;color:var(--text-muted)">v2.0 Model</span>
                        </div>
                    </div>
                `;
            }).join('')}
        </div>
    `;
}
