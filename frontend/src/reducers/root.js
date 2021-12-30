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
    toggleSessionRecording: (state, { payload }) => {
      state.sessionRecording = !state.sessionRecording;
      state.selectedUser = null;
      if (payload) {
        state.customInteractions = payload;
      }
    },
    setSelectedUser: (state, { payload }) => {
      state.selectedUser = payload;
    },
    addUserToFavourites: (state, { payload }) => {
      state.favouriteUsers.push(payload);
    },
    removeUserFromFavourites: (state, { payload }) => {
      state.favouriteUsers = state.favouriteUsers.filter((user) => user !== payload);
    },
    setCustomInteractions: (state, { payload }) => {
      state.customInteractions = payload;
    },
    addCustomInteraction: (state, { payload }) => {
      state.customInteractions.unshift(payload);
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
