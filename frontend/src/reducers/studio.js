/* eslint-disable no-param-reassign */
import { createSlice } from '@reduxjs/toolkit';

export const slice = createSlice({
  name: 'studio',
  initialState: {
    buildMode: true,
    snackbarOpen: false,
    snackbarMessage: '',
    selectedUser: null,
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
    setSelectedUser: (state, { payload }) => {
      state.selectedUser = payload;
    },
  },
});

export const { toggleBuildMode, openSnackbar, closeSnackbar, setSelectedUser } = slice.actions;

export const buildModeSelector = (state) => state.studio.buildMode;

export const snackbarOpenSelector = (state) => state.studio.snackbarOpen;

export const snackbarMessageSelector = (state) => state.studio.snackbarMessage;

export const selectedUserSelector = (state) => state.studio.selectedUser;

export default slice.reducer;
