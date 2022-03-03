import React, { useMemo } from 'react';
import { Routes, Route } from 'react-router-dom';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import { useSelector } from 'react-redux';
import * as colors from '@mui/material/colors';
import CssBaseline from '@mui/material/CssBaseline';

import Layout from './Layout';
import { ModelsEvaluation } from './models';
import { RecPreviews } from './recoms';
import { DatasetEvaluation } from './dataset';
import { SettingsDialog } from './settings';
import Snackbar from './Snackbar';
import { darkModeSelector } from '../reducers/settings';

function App() {
  const darkMode = useSelector(darkModeSelector);

  const theme = useMemo(() => {
    let palette;
    if (!darkMode) {
      palette = {
        primary: {
          main: '#121212',
        },
        background: {
          default: '#fafafa',
        },
      };
    } else {
      palette = {
        mode: 'dark',
      };
    }

    return createTheme({
      palette: {
        ...palette,
        secondary: colors.amber,
      },
    });
  }, [darkMode]);

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Layout>
        <Routes>
          <Route path="/" element={<RecPreviews />} />
          <Route path="/models" element={<ModelsEvaluation />} />
          <Route path="/dataset" element={<DatasetEvaluation />} />
        </Routes>
        <Snackbar />
        <SettingsDialog />
      </Layout>
    </ThemeProvider>
  );
}

export default App;
