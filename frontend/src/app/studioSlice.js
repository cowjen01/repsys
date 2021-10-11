/* eslint-disable no-param-reassign */
import { createSlice } from '@reduxjs/toolkit';

export const studioSlice = createSlice({
  name: 'studio',
  initialState: {
    buildMode: true,
    snackbarOpen: false,
    snackbarMessage: '',
  },
  reducers: {
    toggleBuildMode: (state) => {
      state.buildMode = !state.buildMode;
    },
    openSnackbar: (state, { payload }) => {
      state.snackbarMessage = payload;
      state.snackbarOpen = true;
    },
    closeSnackbar: (state) => {
      state.snackbarOpen = false;
    },
  },
});

export const { toggleBuildMode, openSnackbar, closeSnackbar } = studioSlice.actions;

export const buildModeSelector = (state) => state.studio.buildMode;

export const snackbarOpenSelector = (state) => state.studio.snackbarOpen;

export const snackbarMessageSelector = (state) => state.studio.snackbarMessage;

export default studioSlice.reducer;
