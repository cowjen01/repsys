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
    },
    itemDetailDialog: {
      open: false,
      title: '',
      content: '',
    },
  },
  reducers: {
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
  },
});

export const {
  openSettingsDialog,
  closeSettingsDialog,
  openConfirmDialog,
  closeConfirmDialog,
  openItemDetailDialog,
  closeItemDetailDialog,
} = slice.actions;

export const settingsDialogOpenSelector = (state) => state.dialogs.settingsDialogOpen;

export const confirmDialogSelector = (state) => state.dialogs.confirmDialog;

export const itemDetailDialogSelector = (state) => state.dialogs.itemDetailDialog;

export default slice.reducer;
