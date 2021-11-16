import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import axios from 'axios';

import { requestStateHandler } from './utils';

export const fetchModels = createAsyncThunk('models/fetchModels', async () => {
  const response = await axios.get('/api/models');
  return response.data;
});

export const slice = createSlice({
  name: 'models',
  initialState: {
    data: [],
    status: 'idle',
    error: null,
  },
  extraReducers: requestStateHandler(fetchModels),
});

export const modelsSelector = (state) => state.models.data;
export const modelsStatusSelector = (state) => state.models.status;

export default slice.reducer;
