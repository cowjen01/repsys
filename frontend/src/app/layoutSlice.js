/* eslint-disable no-param-reassign */
import { createSlice } from '@reduxjs/toolkit';
import { nanoid } from 'nanoid';

export const layoutSlice = createSlice({
  name: 'layout',
  initialState: [
    {
      id: 'ajk432j4',
      title: 'Recommended for you',
      model: 'knn-5',
      itemsPerPage: 4,
      totalItems: 20,
    },
    {
      id: 'n392n3m23',
      title: 'Top 4 in Czech Republic',
      model: 'knn-8',
      itemsPerPage: 3,
      totalItems: 20,
    },
  ],
  reducers: {
    addBar: {
      reducer(state, action) {
        state.push(action.payload);
      },
      prepare(title, itemsPerPage) {
        return {
          payload: {
            id: nanoid(),
            title,
            itemsPerPage,
            totalItems: 20,
          },
        };
      },
    },
    updateBarsOrder: (state, action) => {
      const { dragIndex, hoverIndex } = action.payload;
      [state[dragIndex], state[hoverIndex]] = [state[hoverIndex], state[dragIndex]];
    },
    removeBar: (state, action) => {
      state.splice(action.payload, 1);
    },
  },
});

export const { addBar, removeBar, updateBarsOrder } = layoutSlice.actions;

export const layoutSelector = (state) => state.layout;

export default layoutSlice.reducer;
