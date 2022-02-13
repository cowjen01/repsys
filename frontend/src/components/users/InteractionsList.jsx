import React from 'react';
import { Paper, Skeleton, Typography } from '@mui/material';
import { useSelector } from 'react-redux';
import { FixedSizeList } from 'react-window';

import { ItemListView } from '../items';
import { customInteractionsSelector, selectedUserSelector } from '../../reducers/app';

import { useGetUserByIDQuery } from '../../api';

const listHeight = 360;

function renderRow({ index, style, data }) {
  const item = data[index];
  return <ItemListView style={style} key={item.id} item={item} />;
}

function InteractionsList() {
  const customInteractions = useSelector(customInteractionsSelector);
  const selectedUser = useSelector(selectedUserSelector);

  const user = useGetUserByIDQuery(selectedUser, {
    skip: !selectedUser,
  });

  if (selectedUser && user.isLoading) {
    return <Skeleton variant="rectangular" height={listHeight + 48} width="100%" />;
  }

  const interactions = customInteractions.length ? customInteractions : user.data.interactions;

  return (
    <Paper>
      <Typography
        sx={{
          lineHeight: '48px',
          color: 'text.secondary',
          pl: '16px',
        }}
        variant="subtitle2"
        component="div"
      >
        Interactions ({interactions.length})
      </Typography>
      <FixedSizeList
        height={listHeight}
        itemData={interactions}
        itemSize={60}
        itemCount={interactions.length}
      >
        {renderRow}
      </FixedSizeList>
    </Paper>
  );
}

export default InteractionsList;
