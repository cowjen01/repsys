import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import axios from 'axios';

import { requestStateHandler } from './utils';

export const fetchUsers = createAsyncThunk(
  'users/fetchUsers',
  async () => {
    const response = await axios.get('/api/users');
    return response.data;
  }
);

export const slice = createSlice({
  name: 'users',
  initialState: {
    data: [],
    status: 'idle',
    error: null,
  },
  extraReducers: requestStateHandler(fetchUsers),
});

export const usersSelector = (state) => state.users.data;
export const usersStatusSelector = (state) => state.users.status;

export default slice.reducer;
