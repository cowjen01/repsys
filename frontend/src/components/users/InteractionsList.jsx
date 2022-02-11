import React from 'react';
import { Paper, Skeleton, Typography } from '@mui/material';
import { useSelector } from 'react-redux';
import { FixedSizeList } from 'react-window';

import { ItemListView } from '../items';
import { customInteractionsSelector, selectedUserSelector } from '../../reducers/app';
import { itemFieldsSelector } from '../../reducers/settings';

import { useGetInteractionsByUserQuery } from '../../api';

const listHeight = 360;

function renderRow({ index, style, data }) {
  const { interactions, itemFields } = data;
  const item = interactions[index];
  return (
    <ItemListView
      style={style}
      key={item.id}
      id={item.id}
      title={item[itemFields.title]}
      subtitle={item[itemFields.subtitle]}
      image={item[itemFields.image]}
    />
  );
}

function InteractionsList() {
  const customInteractions = useSelector(customInteractionsSelector);
  const selectedUser = useSelector(selectedUserSelector);
  const itemFields = useSelector(itemFieldsSelector);

  const userInteractions = useGetInteractionsByUserQuery(selectedUser, {
    skip: !selectedUser,
  });

  const interactions = customInteractions.length ? customInteractions : userInteractions.data;

  if (selectedUser && userInteractions.isLoading) {
    return <Skeleton variant="rectangular" height={listHeight + 48} width="100%" />;
  }

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
        itemData={{
          interactions,
          itemFields,
        }}
        itemSize={60}
        itemCount={interactions.length}
      >
        {renderRow}
      </FixedSizeList>
    </Paper>
  );
}

export default InteractionsList;
