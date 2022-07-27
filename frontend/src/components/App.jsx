import React, { useMemo, useEffect } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import { useSelector, useDispatch } from 'react-redux';
import * as colors from '@mui/material/colors';
import CssBaseline from '@mui/material/CssBaseline';

import Layout from './Layout';
import { ModelsEvaluation } from './models';
import { RecPreviews } from './recoms';
import { DatasetEvaluation } from './dataset';
import SettingsDialog from './SettingsDialog';
import TutorialDialog from './TutorialDialog';
import Snackbar from './Snackbar';
import { darkModeSelector, setItemView } from '../reducers/settings';
import { useGetDefaultConfigQuery } from '../api';
import { initializedSelector, setInitialized } from '../reducers/app';
import { setRecommenders } from '../reducers/recommenders';
import { DatasetWidget } from './widgets';

function App() {
  const darkMode = useSelector(darkModeSelector);
  const dispatch = useDispatch();
  const appInitialized = useSelector(initializedSelector);
  const defaultConfig = useGetDefaultConfigQuery(
    {},
    {
      skip: appInitialized,
    }
  );

  useEffect(() => {
    if (defaultConfig.isSuccess && !appInitialized) {
      if (defaultConfig.data.recommenders) {
        dispatch(setRecommenders(defaultConfig.data.recommenders));
      }
      if (defaultConfig.data.mappings) {
        dispatch(setItemView(defaultConfig.data.mappings));
      }
      dispatch(setInitialized());
    }
  }, [defaultConfig.isLoading]);

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

  if (!appInitialized) {
    return null;
  }

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<RecPreviews />} />
          <Route path="models" element={<ModelsEvaluation />} />
          <Route path="dataset" element={<DatasetEvaluation />} />
        </Route>
        <Route path="widgets">
          <Route path="dataset/:dataType" element={<DatasetWidget />} />
        </Route>
      </Routes>
      <Snackbar />
      <SettingsDialog />
      <TutorialDialog />
    </ThemeProvider>
  );
}

export default App;
