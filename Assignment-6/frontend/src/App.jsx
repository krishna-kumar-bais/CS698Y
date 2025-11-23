import React, { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route, Link, useLocation } from 'react-router-dom';
import {
  Container,
  Typography,
  Paper,
  Grid,
  TextField,
  Button,
  Card,
  CardContent,
  Box,
  Chip,
  Alert,
  CircularProgress,
  Divider,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Tabs,
  Tab,
} from '@mui/material';
import {
  Calculate,
  Info,
  TrendingUp,
  Warning,
  CheckCircle,
  ExpandMore,
  Assessment,
  Group,
  Timeline,
  Dashboard,
} from '@mui/icons-material';
import axios from 'axios';
import ExplainPanel from './components/ExplainPanel';
import ExplainDashboard from './pages/ExplainDashboard';

// Use environment variable or default to localhost
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000/api';

function Navigation() {
  const location = useLocation();
  const [value, setValue] = useState(location.pathname === '/dashboard' ? 1 : 0);

  useEffect(() => {
    setValue(location.pathname === '/dashboard' ? 1 : 0);
  }, [location]);

  return (
    <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
      <Tabs value={value} aria-label="navigation tabs">
        <Tab 
          label="Predict" 
          icon={<Calculate />} 
          component={Link} 
          to="/" 
          value={0}
        />
        <Tab 
          label="Explainability Dashboard" 
          icon={<Dashboard />} 
          component={Link} 
          to="/dashboard" 
          value={1}
        />
      </Tabs>
    </Box>
  );
}

function PredictPage() {
  // Form state uses friendly keys; we'll map to backend field names on submit
  const [formData, setFormData] = useState({
    age: 35,
    education: 2,
    serviceTime: 5,
    workLoad: 270.0, // matches 'Work load Average/day ' - work units not hours
    transportExpense: 200,
    distance: 10,
    socialDrinker: 0,
    socialSmoker: 0,
    pet: 1,
    son: 1,
    hitTarget: 1,
    month: 6,
    dayOfWeek: 2,
    season: 2,
    reason: 0,
    disciplinaryFailure: 0,
  });

  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [modelInfo, setModelInfo] = useState(null);
  const [featureImportance, setFeatureImportance] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadModelInfo();
    loadFeatureImportance();
  }, []);

  const loadModelInfo = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/model_info`);
      setModelInfo(response.data);
    } catch (err) {
      console.error('Error loading model info:', err);
    }
  };

  const loadFeatureImportance = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/feature_importance`);
      setFeatureImportance(response.data || []);
    } catch (err) {
      // ignore if model not loaded yet
    }
  };

  const handleInputChange = (field, value) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    // Map UI form fields to backend expected keys
    const payload = {
      'Age': formData.age,
      'Education': formData.education,
      'Service time': formData.serviceTime,
      'Work load Average/day ': formData.workLoad,
      'Transportation expense': formData.transportExpense,
      'Distance from Residence to Work': formData.distance,
      'Social drinker': formData.socialDrinker,
      'Social smoker': formData.socialSmoker,
      'Pet': formData.pet,
      'Son': formData.son,
      'Hit target': formData.hitTarget,
      'Month of absence': formData.month,
      'Day of the week': formData.dayOfWeek,
      'Seasons': formData.season,
      'Reason for absence': formData.reason,
      'Disciplinary failure': formData.disciplinaryFailure
    };

    try {
      const response = await axios.post(`${API_BASE_URL}/predict`, payload);
      setPrediction(response.data);
    } catch (err) {
      setError(err.response?.data?.error || 'Error making prediction');
    } finally {
      setLoading(false);
    }
  };

  const getFairnessColor = (gap) => {
    if (gap <= 5) return 'success';
    if (gap <= 15) return 'warning';
    return 'error';
  };

  const getFairnessLabel = (gap) => {
    if (gap <= 5) return 'Good';
    if (gap <= 15) return 'Moderate';
    return 'Poor';
  };

  const handlePredictionUpdate = (newPrediction) => {
    if (prediction) {
      setPrediction({ ...prediction, prediction: newPrediction });
    }
  };

  return (
    <Container maxWidth="md" sx={{ py: 6 }}>
      {/* Header */}
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
          <Calculate sx={{ mr: 2, verticalAlign: 'middle' }} />
          Absenteeism Prediction System
        </Typography>
        <Typography variant="h6" sx={{ opacity: 0.9 }}>
          Predict employee absenteeism using machine learning with fairness considerations
        </Typography>
      </Box>

      <Grid container spacing={4} sx={{ px: 2 }}>
        {/* Input Form */}
        <Grid item xs={12} md={12}>
          <Paper elevation={2} sx={{ p: 3, width: '100%' }}>
            <Typography variant="h5" gutterBottom>
              <Assessment sx={{ mr: 1, verticalAlign: 'middle' }} />
              Employee Information
            </Typography>
            
            <form onSubmit={handleSubmit}>
              {/* Demographics */}
              <Typography variant="h6" color="primary" sx={{ mt: 3, mb: 2 }}>
                Demographics
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <Typography variant="body2" sx={{ fontWeight: 600, mb: 0.5 }}>Age</Typography>
                  <TextField
                    fullWidth
                    placeholder="e.g., 35"
                    type="number"
                    value={formData.age}
                    onChange={(e) => handleInputChange('age', parseInt(e.target.value))}
                    inputProps={{ min: 18, max: 70 }}
                    required
                  />
                </Grid>
                <Grid item xs={12} md={6}>
                  <Typography variant="body2" sx={{ fontWeight: 600, mb: 0.5 }}>Education Level</Typography>
                  <TextField
                    fullWidth
                    select
                    value={formData.education}
                    onChange={(e) => handleInputChange('education', parseInt(e.target.value))}
                    SelectProps={{ native: true }}
                    required
                  >
                    <option value={1}>High School</option>
                    <option value={2}>Graduate</option>
                    <option value={3}>Post Graduate</option>
                    <option value={4}>Master and Doctor</option>
                  </TextField>
                </Grid>
              </Grid>

              <Divider sx={{ my: 3 }} />

              {/* Work Information */}
              <Typography variant="h6" color="primary" sx={{ mt: 3, mb: 2 }}>
                Work Information
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <Typography variant="body2" sx={{ fontWeight: 600, mb: 0.5 }}>Service Time (years)</Typography>
                  <TextField
                    fullWidth
                    placeholder="e.g., 5"
                    type="number"
                    value={formData.serviceTime}
                    onChange={(e) => handleInputChange('serviceTime', parseInt(e.target.value))}
                    inputProps={{ min: 0, max: 30 }}
                    required
                  />
                </Grid>
                <Grid item xs={12} md={6}>
                  <Typography variant="body2" sx={{ fontWeight: 600, mb: 0.5 }}>Work Load (work units/day)</Typography>
                  <TextField
                    fullWidth
                    placeholder="e.g., 270"
                    type="number"
                    value={formData.workLoad}
                    onChange={(e) => handleInputChange('workLoad', parseFloat(e.target.value))}
                    inputProps={{ min: 200, max: 350, step: 1 }}
                    required
                    helperText="Workload intensity (200-350 units)"
                  />
                </Grid>
                <Grid item xs={12} md={6}>
                  <Typography variant="body2" sx={{ fontWeight: 600, mb: 0.5 }}>Transportation Expense</Typography>
                  <TextField
                    fullWidth
                    placeholder="e.g., 200"
                    type="number"
                    value={formData.transportExpense}
                    onChange={(e) => handleInputChange('transportExpense', parseInt(e.target.value))}
                    inputProps={{ min: 0, max: 10000 }}
                    required
                  />
                </Grid>
                <Grid item xs={12} md={6}>
                  <Typography variant="body2" sx={{ fontWeight: 600, mb: 0.5 }}>Distance to Work (km)</Typography>
                  <TextField
                    fullWidth
                    placeholder="e.g., 10"
                    type="number"
                    value={formData.distance}
                    onChange={(e) => handleInputChange('distance', parseFloat(e.target.value))}
                    inputProps={{ min: 0, max: 200, step: 0.1 }}
                    required
                  />
                </Grid>
              </Grid>

              <Divider sx={{ my: 3 }} />

              {/* Behavioral Factors */}
              <Typography variant="h6" color="primary" sx={{ mt: 3, mb: 2 }}>
                Behavioral Factors
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} md={4}>
                  <Typography variant="body2" sx={{ fontWeight: 600, mb: 0.5 }}>Social Drinker</Typography>
                  <TextField
                    fullWidth
                    select
                    value={formData.socialDrinker}
                    onChange={(e) => handleInputChange('socialDrinker', parseInt(e.target.value))}
                    SelectProps={{ native: true }}
                    required
                  >
                    <option value={0}>No</option>
                    <option value={1}>Yes</option>
                  </TextField>
                </Grid>
                <Grid item xs={12} md={4}>
                  <Typography variant="body2" sx={{ fontWeight: 600, mb: 0.5 }}>Social Smoker</Typography>
                  <TextField
                    fullWidth
                    select
                    value={formData.socialSmoker}
                    onChange={(e) => handleInputChange('socialSmoker', parseInt(e.target.value))}
                    SelectProps={{ native: true }}
                    required
                  >
                    <option value={0}>No</option>
                    <option value={1}>Yes</option>
                  </TextField>
                </Grid>
                <Grid item xs={12} md={4}>
                  <Typography variant="body2" sx={{ fontWeight: 600, mb: 0.5 }}>Pet Owner</Typography>
                  <TextField
                    fullWidth
                    select
                    value={formData.pet}
                    onChange={(e) => handleInputChange('pet', parseInt(e.target.value))}
                    SelectProps={{ native: true }}
                    required
                  >
                    <option value={0}>No</option>
                    <option value={1}>Yes</option>
                  </TextField>
                </Grid>
              </Grid>

              <Divider sx={{ my: 3 }} />

              {/* Family Information */}
              <Typography variant="h6" color="primary" sx={{ mt: 3, mb: 2 }}>
                Family Information
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <Typography variant="body2" sx={{ fontWeight: 600, mb: 0.5 }}>Number of Children</Typography>
                  <TextField
                    fullWidth
                    placeholder="e.g., 1"
                    type="number"
                    value={formData.son}
                    onChange={(e) => handleInputChange('son', parseInt(e.target.value))}
                    inputProps={{ min: 0, max: 10 }}
                    required
                  />
                </Grid>
                <Grid item xs={12} md={6}>
                  <Typography variant="body2" sx={{ fontWeight: 600, mb: 0.5 }}>Hit Target</Typography>
                  <TextField
                    fullWidth
                    select
                    value={formData.hitTarget}
                    onChange={(e) => handleInputChange('hitTarget', parseInt(e.target.value))}
                    SelectProps={{ native: true }}
                    required
                  >
                    <option value={0}>No</option>
                    <option value={1}>Yes</option>
                  </TextField>
                </Grid>
              </Grid>

              <Divider sx={{ my: 3 }} />

              {/* Temporal Factors */}
              <Typography variant="h6" color="primary" sx={{ mt: 3, mb: 2 }}>
                Temporal Factors
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} md={4}>
                  <Typography variant="body2" sx={{ fontWeight: 600, mb: 0.5 }}>Month of Absence</Typography>
                  <TextField
                    fullWidth
                    select
                    value={formData.month}
                    onChange={(e) => handleInputChange('month', parseInt(e.target.value))}
                    SelectProps={{ native: true }}
                    required
                  >
                    {[
                      'January', 'February', 'March', 'April', 'May', 'June',
                      'July', 'August', 'September', 'October', 'November', 'December'
                    ].map((month, index) => (
                      <option key={index} value={index + 1}>{month}</option>
                    ))}
                  </TextField>
                </Grid>
                <Grid item xs={12} md={4}>
                  <Typography variant="body2" sx={{ fontWeight: 600, mb: 0.5 }}>Day of Week</Typography>
                  <TextField
                    fullWidth
                    select
                    value={formData.dayOfWeek}
                    onChange={(e) => handleInputChange('dayOfWeek', parseInt(e.target.value))}
                    SelectProps={{ native: true }}
                    required
                  >
                    <option value={2}>Monday</option>
                    <option value={3}>Tuesday</option>
                    <option value={4}>Wednesday</option>
                    <option value={5}>Thursday</option>
                    <option value={6}>Friday</option>
                  </TextField>
                </Grid>
                <Grid item xs={12} md={4}>
                  <Typography variant="body2" sx={{ fontWeight: 600, mb: 0.5 }}>Season</Typography>
                  <TextField
                    fullWidth
                    select
                    value={formData.season}
                    onChange={(e) => handleInputChange('season', parseInt(e.target.value))}
                    SelectProps={{ native: true }}
                    required
                  >
                    <option value={1}>Spring</option>
                    <option value={2}>Summer</option>
                    <option value={3}>Fall</option>
                    <option value={4}>Winter</option>
                  </TextField>
                </Grid>
              </Grid>

              <Divider sx={{ my: 3 }} />

              {/* Additional Factors */}
              <Typography variant="h6" color="primary" sx={{ mt: 3, mb: 2 }}>
                Additional Factors
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} md={8}>
                  <Typography variant="body2" sx={{ fontWeight: 600, mb: 0.5 }}>Reason for Absence</Typography>
                  <TextField
                    fullWidth
                    select
                    value={formData.reason}
                    onChange={(e) => handleInputChange('reason', parseInt(e.target.value))}
                    SelectProps={{ native: true }}
                    required
                  >
                    <option value={0}>No absence</option>
                    <option value={1}>Certain infectious and parasitic diseases</option>
                    <option value={2}>Neoplasms</option>
                    <option value={3}>Diseases of the blood</option>
                    <option value={4}>Endocrine, nutritional and metabolic diseases</option>
                    <option value={5}>Mental and behavioural disorders</option>
                    <option value={6}>Diseases of the nervous system</option>
                    <option value={7}>Diseases of the eye and adnexa</option>
                    <option value={8}>Diseases of the ear and mastoid process</option>
                    <option value={9}>Diseases of the circulatory system</option>
                    <option value={10}>Diseases of the respiratory system</option>
                    <option value={11}>Diseases of the digestive system</option>
                    <option value={12}>Diseases of the skin and subcutaneous tissue</option>
                    <option value={13}>Diseases of the musculoskeletal system and connective tissue</option>
                    <option value={14}>Diseases of the genitourinary system</option>
                    <option value={15}>Pregnancy, childbirth and the puerperium</option>
                    <option value={16}>Certain conditions originating in the perinatal period</option>
                    <option value={17}>Congenital malformations, deformations and chromosomal abnormalities</option>
                    <option value={18}>Abnormal clinical and laboratory findings, not elsewhere classified</option>
                    <option value={19}>Injury, poisoning and certain other consequences of external causes</option>
                    <option value={20}>External causes of morbidity and mortality</option>
                    <option value={21}>Factors influencing health status and contact with health services</option>
                    <option value={22}>Patient follow-up</option>
                    <option value={23}>Medical consultation</option>
                    <option value={24}>Blood donation</option>
                    <option value={25}>Laboratory examination</option>
                    <option value={26}>Unjustified absence</option>
                    <option value={27}>Physiotherapy</option>
                    <option value={28}>Dental consultation</option>
                  </TextField>
                </Grid>
                <Grid item xs={12} md={4}>
                  <Typography variant="body2" sx={{ fontWeight: 600, mb: 0.5 }}>Disciplinary Failure</Typography>
                  <TextField
                    fullWidth
                    select
                    value={formData.disciplinaryFailure}
                    onChange={(e) => handleInputChange('disciplinaryFailure', parseInt(e.target.value))}
                    SelectProps={{ native: true }}
                    required
                  >
                    <option value={0}>No</option>
                    <option value={1}>Yes</option>
                  </TextField>
                </Grid>
              </Grid>

              <Button
                type="submit"
                variant="contained"
                size="large"
                fullWidth
                sx={{
                  mt: 4,
                  py: 1.5,
                  background: 'linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%)',
                  color: 'common.white',
                  boxShadow: 3,
                  '&:hover': {
                    background: 'linear-gradient(135deg, #4338ca 0%, #6d28d9 100%)'
                  }
                }}
                disabled={loading}
                startIcon={loading ? <CircularProgress size={20} /> : <Calculate />}
              >
                {loading ? 'Making Prediction...' : 'Predict Absenteeism'}
              </Button>
            </form>
          </Paper>
        </Grid>

        {/* Results and Model Info */}
        <Grid item xs={12} md={12}>
          {/* Prediction Results */}
          {prediction && (
            <Card sx={{ mb: 3, background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', color: 'white' }}>
              <CardContent sx={{ textAlign: 'center' }}>
                <Typography variant="h5" gutterBottom>
                  <TrendingUp sx={{ mr: 1, verticalAlign: 'middle' }} />
                  Prediction Result
                </Typography>
                <Typography variant="h2" component="div" sx={{ fontWeight: 'bold', mb: 1 }}>
                  {prediction.prediction?.toFixed(2) || '0.00'}
                </Typography>
                <Typography variant="h6" sx={{ mb: 2 }}>
                  hours of predicted absenteeism
                </Typography>
                <Chip
                  label={`Confidence: ${Math.round((prediction.confidence || 0.8) * 100)}%`}
                  sx={{ bgcolor: 'rgba(255,255,255,0.2)', color: 'white' }}
                />
                <Typography variant="body2" sx={{ mt: 1, opacity: 0.9 }}>
                  Use responsibly. This is an estimate based on patterns in historical data.
                </Typography>
              </CardContent>
            </Card>
          )}

          {/* Error Display */}
          {error && (
            <Alert severity="error" sx={{ mb: 3 }}>
              {error}
            </Alert>
          )}

          {/* Explain Panel - Integrated here after prediction */}
          {prediction && (
            <ExplainPanel 
              inputValues={formData} 
              onPredictionUpdate={handlePredictionUpdate}
            />
          )}

          {/* Model Information */}
          <Paper elevation={2} sx={{ p: 3, width: '100%', mt: 3 }}>
            <Typography variant="h5" gutterBottom>
              <Info sx={{ mr: 1, verticalAlign: 'middle' }} />
              Model Information
            </Typography>

            {modelInfo && (
              <>
                {/* Performance Metrics */}
                <Typography variant="h6" sx={{ mt: 3, mb: 2 }}>
                  Model Performance (Baseline vs Mitigated)
                </Typography>
                <Grid container spacing={2} sx={{ mb: 3 }}>
                  {['baseline', 'mitigated'].map((k) => (
                    <Grid item xs={12} md={6} key={k}>
                      <Box sx={{ p: 2, bgcolor: 'grey.50', borderRadius: 2 }}>
                        <Typography variant="subtitle1" sx={{ mb: 1 }}>
                          {k === 'baseline' ? 'Baseline' : 'After Mitigation'}
                        </Typography>
                        <Grid container spacing={1}>
                          <Grid item xs={4}>
                            <Box sx={{ textAlign: 'center' }}>
                              <Typography variant="h6" color="primary">
                                {modelInfo.performance?.[k]?.rmse?.toFixed(2) || 'N/A'}
                              </Typography>
                              <Typography variant="caption" color="text.secondary">RMSE</Typography>
                            </Box>
                          </Grid>
                          <Grid item xs={4}>
                            <Box sx={{ textAlign: 'center' }}>
                              <Typography variant="h6" color="primary">
                                {modelInfo.performance?.[k]?.mae?.toFixed(2) || 'N/A'}
                              </Typography>
                              <Typography variant="caption" color="text.secondary">MAE</Typography>
                            </Box>
                          </Grid>
                          <Grid item xs={4}>
                            <Box sx={{ textAlign: 'center' }}>
                              <Typography variant="h6" color="primary">
                                {modelInfo.performance?.[k]?.r2_score?.toFixed(2) || 'N/A'}
                              </Typography>
                              <Typography variant="caption" color="text.secondary">R²</Typography>
                            </Box>
                          </Grid>
                        </Grid>
                      </Box>
                    </Grid>
                  ))}
                </Grid>

                {/* Fairness Indicators */}
                <Typography variant="h6" sx={{ mt: 3, mb: 2 }}>
                  <Group sx={{ mr: 1, verticalAlign: 'middle' }} />
                  Fairness Indicators
                </Typography>
                {modelInfo.fairness_metrics && (
                  <Grid container spacing={2} sx={{ mb: 3 }}>
                    {['baseline', 'mitigated'].map((k) => (
                      <Grid item xs={12} md={6} key={k}>
                        <Box sx={{ p: 2, bgcolor: 'grey.50', borderRadius: 2 }}>
                          <Typography variant="subtitle1" sx={{ mb: 1 }}>
                            {k === 'baseline' ? 'Baseline' : 'After Mitigation'}
                          </Typography>
                          {Object.entries(modelInfo.fairness_metrics[k]).map(([metricKey, val]) => (
                            <Box key={metricKey} sx={{ mb: 1, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                              <Typography variant="body2">
                                {metricKey.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                              </Typography>
                              <Chip label={`${getFairnessLabel(val)} (${val}h)`} color={getFairnessColor(val)} size="small" />
                            </Box>
                          ))}
                        </Box>
                      </Grid>
                    ))}
                  </Grid>
                )}

                {/* Top Features */}
                {featureImportance?.length > 0 && (
                  <>
                    <Typography variant="h6" sx={{ mt: 3, mb: 1 }}>
                      Most Influential Features (|coefficient|)
                    </Typography>
                    <TableContainer component={Paper} sx={{ mb: 2 }}>
                      <Table size="small">
                        <TableHead>
                          <TableRow>
                            <TableCell>Feature</TableCell>
                            <TableCell align="right">Importance</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {featureImportance.slice(0, 10).map((row, idx) => (
                            <TableRow key={idx}>
                              <TableCell>{row.feature}</TableCell>
                              <TableCell align="right">{(row.importance || 0).toFixed(4)}</TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </TableContainer>
                  </>
                )}

                {/* Bias Mitigation */}
                <Accordion>
                  <AccordionSummary expandIcon={<ExpandMore />}>
                    <Typography variant="h6">
                      <CheckCircle sx={{ mr: 1, verticalAlign: 'middle' }} />
                      Bias Mitigation Applied
                    </Typography>
                  </AccordionSummary>
                  <AccordionDetails>
                    <List dense>
                      {modelInfo.bias_mitigation?.measures?.map((measure, index) => (
                        <ListItem key={index}>
                          <ListItemIcon>
                            <CheckCircle color="success" />
                          </ListItemIcon>
                          <ListItemText primary={measure} />
                        </ListItem>
                      ))}
                    </List>
                  </AccordionDetails>
                </Accordion>

                {/* Model Limitations */}
                <Accordion sx={{ mt: 2 }}>
                  <AccordionSummary expandIcon={<ExpandMore />}>
                    <Typography variant="h6">
                      <Warning sx={{ mr: 1, verticalAlign: 'middle' }} />
                      Model Limitations
                    </Typography>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Alert severity="warning" sx={{ mb: 2 }}>
                      <strong>Important:</strong> This model has limited performance due to data imbalance and should be used as a guideline only.
                    </Alert>
                    <List dense>
                      {modelInfo.limitations?.map((limitation, index) => (
                        <ListItem key={index}>
                          <ListItemIcon>
                            <Warning color="warning" />
                          </ListItemIcon>
                          <ListItemText primary={limitation} />
                        </ListItem>
                      ))}
                    </List>
                  </AccordionDetails>
                </Accordion>
              </>
            )}
          </Paper>
        </Grid>
      </Grid>
    </Container>
  );
}

function App() {
  return (
    <BrowserRouter>
      <Container maxWidth="lg" sx={{ py: 2 }}>
        <Navigation />
        <Routes>
          <Route path="/" element={<PredictPage />} />
          <Route path="/dashboard" element={<ExplainDashboard />} />
        </Routes>
      </Container>
    </BrowserRouter>
  );
}

export default App;
