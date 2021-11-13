/* eslint-disable no-param-reassign */
import { createSlice } from '@reduxjs/toolkit';

export const slice = createSlice({
  name: 'studio',
  initialState: {
    buildMode: true,
    snackbarOpen: false,
    snackbarMessage: '',
    selectedUser: null,
    favouriteUsers: [],
    sessionRecord: false,
  },
  reducers: {
    toggleBuildMode: (state) => {
      state.buildMode = !state.buildMode;
    },
    toggleSessionRecord: (state) => {
      state.sessionRecord = !state.sessionRecord;
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
    addUserToFavourites: (state) => {
      state.favouriteUsers.push(state.selectedUser);
    },
    removeUserFromFavourites: (state) => {
      state.favouriteUsers = state.favouriteUsers.filter(
        (user) => user.id !== state.selectedUser.id
      );
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
  toggleSessionRecord,
} = slice.actions;

export const buildModeSelector = (state) => state.studio.buildMode;

export const snackbarOpenSelector = (state) => state.studio.snackbarOpen;

export const snackbarMessageSelector = (state) => state.studio.snackbarMessage;

export const selectedUserSelector = (state) => state.studio.selectedUser;

export const favouriteUsersSelector = (state) => state.studio.favouriteUsers;

export const sessionRecordSelector = (state) => state.studio.sessionRecord;

export default slice.reducer;
