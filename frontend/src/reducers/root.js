/* eslint-disable no-param-reassign */
import { createSlice } from '@reduxjs/toolkit';

export const slice = createSlice({
  name: 'root',
  initialState: {
    buildMode: false,
    selectedUser: null,
    favouriteUsers: [],
    sessionRecording: false,
    customInteractions: [],
  },
  reducers: {
    toggleBuildMode: (state) => {
      state.buildMode = !state.buildMode;
    },
    toggleSessionRecording: (state) => {
      state.sessionRecording = !state.sessionRecording;
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
    setCustomInteractions: (state, { payload }) => {
      state.customInteractions = payload;
    },
    addCustomInteraction: (state, { payload }) => {
      state.customInteractions.push(payload);
    },
  },
});

export const {
  toggleBuildMode,
  setSelectedUser,
  addUserToFavourites,
  removeUserFromFavourites,
  toggleSessionRecording,
  addCustomInteraction,
  setCustomInteractions,
} = slice.actions;

export const buildModeSelector = (state) => state.root.buildMode;

export const selectedUserSelector = (state) => state.root.selectedUser;

export const favouriteUsersSelector = (state) => state.root.favouriteUsers;

export const sessionRecordingSelector = (state) => state.root.sessionRecording;

export const customInteractionsSelector = (state) => state.root.customInteractions;

export default slice.reducer;
