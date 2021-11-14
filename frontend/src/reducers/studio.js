/* eslint-disable no-param-reassign */
import { createSlice } from '@reduxjs/toolkit';

export const slice = createSlice({
  name: 'studio',
  initialState: {
    buildMode: false,
    snackbar: {
      open: false,
      message: '',
      severity: 'success',
    },
    selectedUser: null,
    favouriteUsers: [],
    sessionRecording: false,
    customInteractions: [],
    settingsDialogOpen: false,
    confirmDialog: {
      open: false,
      title: '',
      content: '',
    },
    itemDetailDialog: {
      open: false,
      title: '',
      content: '',
    },
  },
  reducers: {
    toggleBuildMode: (state) => {
      state.buildMode = !state.buildMode;
    },
    toggleSessionRecording: (state) => {
      state.sessionRecording = !state.sessionRecording;
    },
    openSnackbar: (state, { payload }) => {
      state.snackbar = {
        open: true,
        message: payload.message || '',
        severity: payload.severity || 'success',
      };
    },
    closeSnackbar: (state) => {
      state.snackbar.open = false;
    },
    openConfirmDialog: (state, { payload }) => {
      state.confirmDialog = {
        open: true,
        title: payload.title || '',
        content: payload.content || '',
      };
    },
    closeConfirmDialog: (state) => {
      state.confirmDialog.open = false;
    },
    openItemDetailDialog: (state, { payload }) => {
      state.itemDetailDialog = {
        open: true,
        title: payload.title || '',
        content: payload.content || 'No description provided.',
      };
    },
    closeItemDetailDialog: (state) => {
      state.itemDetailDialog.open = false;
    },
    openSettingsDialog: (state) => {
      state.settingsDialogOpen = true;
    },
    closeSettingsDialog: (state) => {
      state.settingsDialogOpen = false;
    },
    setSelectedUser: (state, { payload }) => {
      state.selectedUser = payload;
    },
    addUserToFavourites: (state) => {
      state.favouriteUsers.push(state.selectedUser);
    },
    removeUserFromFavourites: (state) => {
      state.favouriteUsers = state.favouriteUsers.filter(
        (user) => user.id !== state.selectedUser.id
      );
    },
    addCustomInteraction: (state, { payload }) => {
      state.customInteractions.push(payload);
    },
    clearCustomInteractions: (state) => {
      state.customInteractions = [];
    },
  },
});

export const {
  toggleBuildMode,
  openSnackbar,
  closeSnackbar,
  setSelectedUser,
  addUserToFavourites,
  removeUserFromFavourites,
  toggleSessionRecording,
  addCustomInteraction,
  clearCustomInteractions,
  openSettingsDialog,
  closeSettingsDialog,
  openConfirmDialog,
  closeConfirmDialog,
  openItemDetailDialog,
  closeItemDetailDialog,
} = slice.actions;

export const buildModeSelector = (state) => state.studio.buildMode;

export const snackbarSelector = (state) => state.studio.snackbar;

export const selectedUserSelector = (state) => state.studio.selectedUser;

export const favouriteUsersSelector = (state) => state.studio.favouriteUsers;

export const sessionRecordingSelector = (state) => state.studio.sessionRecording;

export const customInteractionsSelector = (state) => state.studio.customInteractions;

export const settingsDialogOpenSelector = (state) => state.studio.settingsDialogOpen;

export const confirmDialogSelector = (state) => state.studio.confirmDialog;

export const itemDetailDialogSelector = (state) => state.studio.itemDetailDialog;

export default slice.reducer;
