/* eslint-disable no-param-reassign */
import { createSlice } from '@reduxjs/toolkit';

export const studioSlice = createSlice({
  name: 'studio',
  initialState: {
    buildMode: true,
    darkMode: false,
    snackbarOpen: false,
    snackbarMessage: '',
    metricsOpen: false
  },
  reducers: {
    toggleBuildMode: (state) => {
      state.buildMode = !state.buildMode;
    },
    toggleDarkMode: (state) => {
      state.darkMode = !state.darkMode;
    },
    openSnackbar: (state, { payload }) => {
      state.snackbarMessage = payload;
      state.snackbarOpen = true;
    },
    closeSnackbar: (state) => {
      state.snackbarOpen = false;
    },
    openModelMetrics: (state) => {
      state.metricsOpen = true;
    },
    closeModelMetrics: (state) => {
      state.metricsOpen = false;
    }
  },
});

export const { toggleBuildMode, openSnackbar, closeSnackbar, toggleDarkMode, openModelMetrics, closeModelMetrics } = studioSlice.actions;

export const buildModeSelector = (state) => state.studio.buildMode;

export const darkModeSelector = (state) => state.studio.darkMode;

export const snackbarOpenSelector = (state) => state.studio.snackbarOpen;

export const snackbarMessageSelector = (state) => state.studio.snackbarMessage;

export const modelMetricsSelector = (state) => state.studio.metricsOpen;

export default studioSlice.reducer;
