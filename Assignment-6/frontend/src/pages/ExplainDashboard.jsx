import React, { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  Paper,
  Box,
  Alert,
  CircularProgress,
  Card,
  CardContent,
  Chip,
  Grid,
} from '@mui/material';
import {
  Assessment,
  Cached,
} from '@mui/icons-material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import axios from 'axios';

// Determine API base URL: prefer env override; else same origin in prod,
// and localhost in dev.
const getApiBaseUrl = () => {
  if (import.meta.env.VITE_API_BASE_URL) {
    return import.meta.env.VITE_API_BASE_URL.replace('/api', '');
  }
  const { hostname, protocol } = window.location;
  if (hostname !== 'localhost' && hostname !== '127.0.0.1') {
    return `${protocol}//${hostname}`;
  }
  return 'http://localhost:5000';
};
const EXPLAIN_API_BASE_URL = getApiBaseUrl();

function ExplainDashboard() {
  const [globalExplanation, setGlobalExplanation] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchGlobalExplanation();
  }, []);

  const fetchGlobalExplanation = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.get(`${EXPLAIN_API_BASE_URL}/explain/global`);
      setGlobalExplanation(response.data);
    } catch (err) {
      // Fallback: use coefficient-based importance from /api/feature_importance
      try {
        const coeffResp = await axios.get(`${EXPLAIN_API_BASE_URL}/api/feature_importance`);
        const feature_importance = (coeffResp.data || []).map((item) => ({
          feature: item.feature,
          mean_abs_shap: item.importance,
        }));
        setGlobalExplanation({
          feature_importance,
          explainer_type: 'CoefficientFallback',
          sample_size: feature_importance.length,
          cached: false,
        });
      } catch (fallbackErr) {
        setError(
          err.response?.data?.error ||
          fallbackErr.response?.data?.error ||
          'Error fetching global explanation'
        );
      }
    } finally {
      setLoading(false);
    }
  };

  const getChartData = () => {
    if (!globalExplanation?.feature_importance) return [];
    
    return globalExplanation.feature_importance
      .slice(0, 15)
      .map((feat) => ({
        feature: feat.feature.length > 25 
          ? feat.feature.substring(0, 25) + '...' 
          : feat.feature,
        fullName: feat.feature,
        importance: feat.mean_abs_shap,
      }))
      .reverse(); // Reverse to show highest at top
  };

  return (
    <Container maxWidth="lg" sx={{ py: 6 }}>
      <Box sx={{
        width: '100%',
        p: { xs: 3, md: 5 },
        mb: 4,
        textAlign: 'center',
        borderRadius: 3,
        color: 'common.white',
        background: 'linear-gradient(135deg, #4f46e5 0%, #8b5cf6 100%)',
        boxShadow: 3,
      }}>
        <Typography variant="h3" component="h1" gutterBottom sx={{ color: 'common.white' }}>
          <Assessment sx={{ mr: 2, verticalAlign: 'middle' }} />
          Explainability Dashboard
        </Typography>
        <Typography variant="h6" sx={{ opacity: 0.9 }}>
          Global Feature Importance Analysis
        </Typography>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {loading ? (
        <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
          <CircularProgress />
        </Box>
      ) : globalExplanation ? (
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                  <Typography variant="h5">
                    Global Feature Importance (SHAP)
                  </Typography>
                  <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                    {globalExplanation.cached && (
                      <Chip 
                        icon={<Cached />}
                        label="Cached" 
                        color="success" 
                        size="small"
                      />
                    )}
                    <Chip 
                      label={globalExplanation.explainer_type}
                      variant="outlined"
                      size="small"
                    />
                    <Chip 
                      label={`n=${globalExplanation.sample_size}`}
                      variant="outlined"
                      size="small"
                    />
                  </Box>
                </Box>
                
                <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                  Mean absolute SHAP values across {globalExplanation.sample_size} samples. 
                  Higher values indicate more influential features.
                </Typography>

                <ResponsiveContainer width="100%" height={500}>
                  <BarChart
                    data={getChartData()}
                    layout="vertical"
                    margin={{ top: 5, right: 30, left: 150, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" />
                    <YAxis 
                      dataKey="feature" 
                      type="category" 
                      width={140}
                      tick={{ fontSize: 12 }}
                    />
                    <Tooltip 
                      formatter={(value) => value.toFixed(4)}
                      labelFormatter={(label, payload) => 
                        payload?.[0]?.payload?.fullName || label
                      }
                    />
                    <Bar 
                      dataKey="importance" 
                      fill="#4f46e5"
                      radius={[0, 8, 8, 0]}
                    />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Top 10 Most Important Features
              </Typography>
              <Box component="ol" sx={{ pl: 3 }}>
                {globalExplanation.feature_importance.slice(0, 10).map((feat, idx) => (
                  <Typography key={idx} component="li" sx={{ mb: 1 }}>
                    <strong>{feat.feature}</strong>: {feat.mean_abs_shap.toFixed(4)}
                  </Typography>
                ))}
              </Box>
            </Paper>
          </Grid>
        </Grid>
      ) : (
        <Alert severity="warning">
          No global explanation data available
        </Alert>
      )}
    </Container>
  );
}

export default ExplainDashboard;

