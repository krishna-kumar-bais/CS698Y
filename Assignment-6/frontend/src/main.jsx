import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { ThemeProvider, createTheme, responsiveFontSizes } from '@mui/material/styles'
import CssBaseline from '@mui/material/CssBaseline'
import App from './App.jsx'

let theme = createTheme({
  palette: {
    mode: 'light',
    primary: { main: '#4f46e5' }, // indigo-600
    secondary: { main: '#8b5cf6' }, // violet-500
    background: {
      default: '#f8fafc',
      paper: '#ffffff'
    }
  },
  shape: { borderRadius: 12 },
  typography: {
    fontFamily: '"Roboto","Helvetica","Arial",sans-serif',
    h3: { fontWeight: 700 },
    h5: { fontWeight: 600 }
  },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        body: {
          background: 'linear-gradient(135deg, #eef2ff 0%, #f5f3ff 100%)',
          minHeight: '100vh'
        }
      }
    },
    MuiPaper: {
      defaultProps: { elevation: 2 },
      styleOverrides: {
        root: {
          borderRadius: 16
        }
      }
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 16
        }
      }
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          textTransform: 'none',
          fontWeight: 600
        }
      }
    },
    MuiTextField: {
      defaultProps: { fullWidth: true, variant: 'outlined', InputLabelProps: { shrink: true } }
    },
    // Ensure long labels aren't truncated inside the notch
    MuiInputLabel: {
      styleOverrides: {
        root: {
          whiteSpace: 'normal',
          overflow: 'visible',
          textOverflow: 'clip',
          maxWidth: 'none'
        }
      }
    }
  }
});

theme = responsiveFontSizes(theme);

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <App />
    </ThemeProvider>
  </StrictMode>,
)
