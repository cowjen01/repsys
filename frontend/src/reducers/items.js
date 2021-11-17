import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import axios from 'axios';

import { requestStateHandler } from './utils';

export const fetchItems = createAsyncThunk(
  'items/fetchItems',
  async (queryString) => {
    const response = await axios.get('/api/items', {
      params: {
        query: queryString,
      },
    });
    return response.data;
  }
);

export const slice = createSlice({
  name: 'items',
  initialState: {
    data: [],
    status: 'idle',
    error: null,
  },
  extraReducers: requestStateHandler(fetchItems),
});

export const itemsSelector = (state) => state.items.data;
export const itemsStatusSelector = (state) => state.items.status;

export default slice.reducer;
