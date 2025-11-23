import React, { useState, useEffect } from 'react';
import {
  Paper,
  Typography,
  Button,
  Box,
  Alert,
  CircularProgress,
  Card,
  CardContent,
  Grid,
  Slider,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  Divider,
} from '@mui/material';
import {
  Lightbulb,
  Psychology,
  AutoFixHigh,
  TrendingUp,
} from '@mui/icons-material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts';
import axios from 'axios';

// Resolve API base URL: prefer explicit env; otherwise use same origin in prod,
// and localhost during development.
const getApiBaseUrl = () => {
  if (import.meta.env.VITE_API_BASE_URL) {
    return import.meta.env.VITE_API_BASE_URL.replace('/api', '');
  }
  const { hostname, protocol } = window.location;
  // In production (Render), use same origin. Ports are managed by the platform.
  if (hostname !== 'localhost' && hostname !== '127.0.0.1') {
    return `${protocol}//${hostname}`;
  }
  // Local development fallback
  return 'http://localhost:5000';
};
const EXPLAIN_API_BASE_URL = getApiBaseUrl();

function ExplainPanel({ inputValues, onPredictionUpdate }) {
  const [localExplanation, setLocalExplanation] = useState(null);
  const [limeExplanation, setLimeExplanation] = useState(null);
  const [counterfactual, setCounterfactual] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('shap'); // 'shap', 'lime', 'cf'
  const [whatIfValues, setWhatIfValues] = useState({
    workLoad: inputValues.workLoad || 270,
    distance: inputValues.distance || 10,
  });

  // Map form field names to backend field names
  const mapToBackendFields = (values) => {
    return {
      'Age': values.age,
      'Education': values.education,
      'Service time': values.serviceTime,
      'Work load Average/day ': values.workLoad || whatIfValues.workLoad,
      'Transportation expense': values.transportExpense,
      'Distance from Residence to Work': values.distance || whatIfValues.distance,
      'Social drinker': values.socialDrinker,
      'Social smoker': values.socialSmoker,
      'Pet': values.pet,
      'Son': values.son,
      'Hit target': values.hitTarget,
      'Month of absence': values.month,
      'Day of the week': values.dayOfWeek,
      'Seasons': values.season,
      'Reason for absence': values.reason,
      'Disciplinary failure': values.disciplinaryFailure,
    };
  };

  const fetchLocalExplanation = async (values = inputValues) => {
    setLoading(true);
    setError(null);
    try {
      const payload = mapToBackendFields(values);
      const response = await axios.post(`${EXPLAIN_API_BASE_URL}/explain/local`, {
        input: payload,
      });
      setLocalExplanation(response.data);
      
      // Update parent prediction if callback provided
      if (onPredictionUpdate) {
        onPredictionUpdate(response.data.prediction);
      }
    } catch (err) {
      setError(err.response?.data?.error || 'Error fetching explanation');
    } finally {
      setLoading(false);
    }
  };

  const fetchLimeExplanation = async (values = inputValues) => {
    setLoading(true);
    setError(null);
    try {
      const payload = mapToBackendFields(values);
      const response = await axios.post(`${EXPLAIN_API_BASE_URL}/explain/lime`, {
        input: payload,
      });
      setLimeExplanation(response.data);
    } catch (err) {
      setError(err.response?.data?.error || 'Error fetching LIME explanation');
    } finally {
      setLoading(false);
    }
  };

  const fetchCounterfactual = async (values = inputValues) => {
    setLoading(true);
    setError(null);
    try {
      const payload = mapToBackendFields(values);
      const response = await axios.post(`${EXPLAIN_API_BASE_URL}/explain/cf`, {
        input: payload,
        target: 0.8,
      });
      setCounterfactual(response.data);
    } catch (err) {
      setError(err.response?.data?.error || 'Error fetching counterfactual');
    } finally {
      setLoading(false);
    }
  };

  const handleWhatIfChange = async (field, value) => {
    const newValues = { ...whatIfValues, [field]: value };
    setWhatIfValues(newValues);

    // Update input values and fetch new explanation
    const updatedValues = { ...inputValues, [field]: value };
    await fetchLocalExplanation(updatedValues);

    // Keep other explanation tabs in sync with What-If adjustments
    if (activeTab === 'lime') {
      await fetchLimeExplanation(updatedValues);
    } else if (activeTab === 'cf') {
      await fetchCounterfactual(updatedValues);
    }
  };

  // Prepare chart data for SHAP
  const getShapChartData = () => {
    if (!localExplanation?.contributions) return [];
    
    return localExplanation.contributions
      .slice(0, 6)
      .map((contrib) => ({
        feature: contrib.feature.length > 20 
          ? contrib.feature.substring(0, 20) + '...' 
          : contrib.feature,
        fullName: contrib.feature,
        shapValue: contrib.shap,
      }))
      .sort((a, b) => Math.abs(b.shapValue) - Math.abs(a.shapValue));
  };

  return (
    <Paper elevation={2} sx={{ p: 3, mt: 3 }}>
      <Typography variant="h5" gutterBottom>
        <Lightbulb sx={{ mr: 1, verticalAlign: 'middle' }} />
        Explanation & What-If Analysis (Local Explanation)
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {/* Action Buttons */}
      <Box sx={{ mb: 3, display: 'flex', gap: 2, flexWrap: 'wrap' }}>
        <Button
          variant={activeTab === 'shap' ? "contained" : "outlined"}
          startIcon={<Psychology />}
          onClick={() => {
            setActiveTab('shap');
            fetchLocalExplanation();
          }}
          disabled={loading}
        >
          {loading && activeTab === 'shap' ? <CircularProgress size={20} sx={{ mr: 1 }} /> : null}
          Explain with SHAP
        </Button>
        <Button
          variant={activeTab === 'lime' ? "contained" : "outlined"}
          startIcon={<AutoFixHigh />}
          onClick={() => {
            setActiveTab('lime');
            // Use latest What-If values when requesting LIME
            const merged = { ...inputValues, workLoad: whatIfValues.workLoad, distance: whatIfValues.distance };
            fetchLimeExplanation(merged);
          }}
          disabled={loading}
        >
          {loading && activeTab === 'lime' ? <CircularProgress size={20} sx={{ mr: 1 }} /> : null}
          Show LIME
        </Button>
        <Button
          variant={activeTab === 'cf' ? "contained" : "outlined"}
          startIcon={<TrendingUp />}
          onClick={() => {
            setActiveTab('cf');
            // Use latest What-If values when requesting Counterfactuals
            const merged = { ...inputValues, workLoad: whatIfValues.workLoad, distance: whatIfValues.distance };
            fetchCounterfactual(merged);
          }}
          disabled={loading}
        >
          {loading && activeTab === 'cf' ? <CircularProgress size={20} sx={{ mr: 1 }} /> : null}
          Show Counterfactual
        </Button>
      </Box>

      {/* What-If Sliders */}
      <Card sx={{ mb: 3, bgcolor: 'grey.50' }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            What-If Analysis
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Adjust these values to see how predictions change:
          </Typography>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Typography gutterBottom>
                Work Load (work units/day): {whatIfValues.workLoad.toFixed(1)}
              </Typography>
              <Slider
                value={whatIfValues.workLoad}
                onChange={(e, value) => handleWhatIfChange('workLoad', value)}
                min={200}
                max={350}
                step={1}
                marks={[
                  { value: 200, label: '200' },
                  { value: 275, label: '275' },
                  { value: 350, label: '350' },
                ]}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography gutterBottom>
                Distance to Work (km): {whatIfValues.distance.toFixed(1)}
              </Typography>
              <Slider
                value={whatIfValues.distance}
                onChange={(e, value) => handleWhatIfChange('distance', value)}
                min={1}
                max={50}
                step={1}
                marks={[
                  { value: 1, label: '1' },
                  { value: 25, label: '25' },
                  { value: 50, label: '50' },
                ]}
              />
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* SHAP Explanation */}
      {activeTab === 'shap' && localExplanation && (
        <Box>
          <Typography variant="h6" gutterBottom>
            SHAP Feature Contributions
          </Typography>
          {localExplanation.text_summary && (
            <Alert severity="info" sx={{ mb: 2 }}>
              {localExplanation.text_summary}
            </Alert>
          )}
          
          <Card sx={{ mb: 2 }}>
            <CardContent>
              <Typography variant="subtitle2" gutterBottom>
                Top 6 Features Contributing to Prediction
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={getShapChartData()} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" />
                  <YAxis dataKey="feature" type="category" width={150} />
                  <Tooltip 
                    formatter={(value) => value.toFixed(4)}
                    labelFormatter={(label, payload) => 
                      payload?.[0]?.payload?.fullName || label
                    }
                  />
                  <Bar 
                    dataKey="shapValue" 
                    fill="#4f46e5"
                    radius={[0, 8, 8, 0]}
                  >
                    {getShapChartData().map((entry, index) => (
                      <Cell 
                        key={`cell-${index}`} 
                        fill={entry.shapValue > 0 ? '#4f46e5' : '#ef4444'} 
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Box>
      )}

      {/* LIME Explanation */}
      {activeTab === 'lime' && limeExplanation && (
        <Box>
          <Typography variant="h6" gutterBottom>
            LIME Explanation
          </Typography>
          <Alert severity="info" sx={{ mb: 2 }}>
            Prediction: {limeExplanation.prediction?.toFixed(2)} hours | 
            Explanation Score: {limeExplanation.explanation_score?.toFixed(2)}
          </Alert>
          
          <TableContainer component={Paper}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell><strong>Feature</strong></TableCell>
                  <TableCell align="right"><strong>Weight</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {limeExplanation.top_features?.slice(0, 10).map((feat, idx) => (
                  <TableRow key={idx}>
                    <TableCell>{feat.feature}</TableCell>
                    <TableCell align="right">
                      <Chip 
                        label={feat.weight.toFixed(4)} 
                        color={feat.weight > 0 ? 'primary' : 'secondary'}
                        size="small"
                      />
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </Box>
      )}

      {/* Counterfactual */}
      {activeTab === 'cf' && counterfactual && (
        <Box>
          <Typography variant="h6" gutterBottom>
            Counterfactual Suggestions
          </Typography>
          <Alert severity="info" sx={{ mb: 2 }}>
            Original: {counterfactual.original_prediction?.toFixed(2)} hours → 
            Target: {counterfactual.target_prediction?.toFixed(2)} hours
          </Alert>
          
          {counterfactual.candidates && counterfactual.candidates.length > 0 ? (
            <TableContainer component={Paper}>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell><strong>Feature</strong></TableCell>
                    <TableCell align="right"><strong>Change</strong></TableCell>
                    <TableCell align="right"><strong>New Prediction</strong></TableCell>
                    <TableCell align="right"><strong>Reduction</strong></TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {counterfactual.candidates.slice(0, 5).map((cand, idx) => (
                    <TableRow key={idx}>
                      <TableCell>{cand.feature}</TableCell>
                      <TableCell align="right">
                        {cand.change > 0 ? '+' : ''}{cand.change.toFixed(2)}
                      </TableCell>
                      <TableCell align="right">{cand.new_prediction.toFixed(2)}h</TableCell>
                      <TableCell align="right">
                        <Chip 
                          label={`${cand.reduction_percent.toFixed(1)}%`}
                          color="success"
                          size="small"
                        />
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          ) : (
            <Alert severity="warning">No counterfactual suggestions available</Alert>
          )}
        </Box>
      )}

      {!localExplanation && !limeExplanation && !counterfactual && !loading && (
        <Alert severity="info">
          Click "Explain with SHAP" to see feature contributions for this prediction
        </Alert>
      )}
    </Paper>
  );
}

export default ExplainPanel;

