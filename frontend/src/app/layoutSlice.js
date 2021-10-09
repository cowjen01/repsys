/* eslint-disable no-param-reassign */
import { createSlice } from '@reduxjs/toolkit';
import { customAlphabet } from 'nanoid';

const nanoid = customAlphabet('1234567890abcdef', 10);

export const layoutSlice = createSlice({
  name: 'layout',
  initialState: [
    {
      id: nanoid(),
      title: 'Recommended for you',
      model: 'knn-5',
      itemsPerPage: 4,
      totalItems: 20,
    },
    {
      id: nanoid(),
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
    duplicateBar: {
      reducer(state, action) {
        const { index, id } = action.payload;
        const source = state[index];
        state.splice(index + 1, 0, {
          ...source,
          id,
          title: `${source.title} - copy`,
        });
      },
      prepare(index) {
        return {
          payload: {
            index,
            id: nanoid(),
          },
        };
      },
    },
    removeBar: (state, action) => {
      state.splice(action.payload, 1);
    },
  },
});

export const { addBar, removeBar, updateBarsOrder, duplicateBar } = layoutSlice.actions;

export const layoutSelector = (state) => state.layout;

export default layoutSlice.reducer;
