/* eslint-disable no-param-reassign */
import { createSlice } from '@reduxjs/toolkit';

export const slice = createSlice({
  name: 'dialogs',
  initialState: {
    settingsDialogOpen: false,
    confirmDialog: {
      open: false,
      title: '',
      content: '',
      params: null,
    },
    itemDetailDialog: {
      open: false,
      item: null,
    },
    snackbar: {
      open: false,
      message: '',
      severity: 'success',
    },
    userSelectDialogOpen: false,
    recEditDialog: {
      open: false,
      index: null,
    },
  },
  reducers: {
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
        params: payload.params,
      };
    },
    closeConfirmDialog: (state) => {
      state.confirmDialog.open = false;
    },
    openItemDetailDialog: (state, { payload }) => {
      state.itemDetailDialog = {
        open: true,
        item: payload,
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
    openUserSelectDialog: (state) => {
      state.userSelectDialogOpen = true;
    },
    closeUserSelectDialog: (state) => {
      state.userSelectDialogOpen = false;
    },
    openRecEditDialog: (state, { payload }) => {
      state.recEditDialog = {
        open: true,
        index: payload,
      };
    },
    closeRecEditDialog: (state) => {
      state.recEditDialog = {
        open: false,
        index: null,
      };
    },
  },
});

export const {
  openSnackbar,
  closeSnackbar,
  openSettingsDialog,
  closeSettingsDialog,
  openConfirmDialog,
  closeConfirmDialog,
  openItemDetailDialog,
  closeItemDetailDialog,
  openUserSelectDialog,
  closeUserSelectDialog,
  openRecEditDialog,
  closeRecEditDialog,
} = slice.actions;

export const snackbarSelector = (state) => state.dialogs.snackbar;

export const settingsDialogSelector = (state) => state.dialogs.settingsDialogOpen;

export const confirmDialogSelector = (state) => state.dialogs.confirmDialog;

export const itemDetailDialogSelector = (state) => state.dialogs.itemDetailDialog;

export const userSelectDialogSelector = (state) => state.dialogs.userSelectDialogOpen;

export const recEditDialogSelector = (state) => state.dialogs.recEditDialog;

export default slice.reducer;
