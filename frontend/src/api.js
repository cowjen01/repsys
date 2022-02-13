import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react';

export const repsysApi = createApi({
  reducerPath: 'repsysApi',
  baseQuery: fetchBaseQuery({ baseUrl: '/api/' }),
  endpoints: (builder) => ({
    getUsers: builder.query({
      query: (split = 'train') => `/users?split=${split}`,
    }),
    searchUsersByInteractions: builder.mutation({
      query: ({ split = 'train', attribute, value, threshold }) => ({
        url: `/users/search`,
        method: 'POST',
        body: {
          split,
          interactions: {
            attribute,
            value,
            threshold,
          },
        },
      }),
    }),
    searchItemsByAttribute: builder.mutation({
      query: ({ attribute, value }) => ({
        url: `/items/search`,
        method: 'POST',
        body: {
          attribute,
          value,
        },
      }),
    }),
    getItemsByTitle: builder.query({
      query: (title) => `/items?query=${title}`,
    }),
    getUserByID: builder.query({
      query: (userID) => `/users/${userID}`,
    }),
    describeUsers: builder.mutation({
      query: ({ split = 'train', users }) => ({
        url: `/users/describe`,
        method: 'POST',
        body: {
          split,
          users,
        },
      }),
    }),
    getDataset: builder.query({
      query: () => '/dataset',
    }),
    getModels: builder.query({
      query: () => '/models',
    }),
    getMetricsByModel: builder.query({
      query: (model) => `/models/${model}/metrics`,
    }),
    predictItemsByModel: builder.mutation({
      query: ({ model, ...body }) => ({
        url: `/models/${model}/predict`,
        method: 'POST',
        body,
      }),
    }),
    getModelsMetrics: builder.query({
      query: () => '/models/metrics',
    }),
    describeItems: builder.mutation({
      query: (items) => ({
        url: '/items/describe',
        method: 'POST',
        body: {
          items,
        },
      }),
    }),
    getUsersEmbeddings: builder.query({
      query: (split = 'train') => `/users/embeddings?split=${split}`,
    }),
    getItemsEmbeddings: builder.query({
      query: (split = 'train') => `/items/embeddings?split=${split}`,
    }),
  }),
});

export const {
  useGetDatasetQuery,
  useGetItemsByTitleQuery,
  useGetItemsEmbeddingsQuery,
  useGetMetricsByModelQuery,
  useGetModelsMetricsQuery,
  useGetModelsQuery,
  useGetUserByIDQuery,
  useGetUsersEmbeddingsQuery,
  useGetUsersQuery,
  useDescribeItemsMutation,
  useDescribeUsersMutation,
  useSearchItemsByAttributeMutation,
  useSearchUsersByInteractionsMutation,
  usePredictItemsByModelMutation,
} = repsysApi;
