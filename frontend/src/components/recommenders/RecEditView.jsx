/* eslint-disable no-param-reassign */
import React, { useRef } from 'react';
import pt from 'prop-types';
import { useDrag, useDrop } from 'react-dnd';
import DragIndicatorIcon from '@mui/icons-material/DragIndicator';
import EditIcon from '@mui/icons-material/Edit';
import DeleteIcon from '@mui/icons-material/Delete';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import { Typography, Paper, Box, IconButton, Stack } from '@mui/material';
import { useDispatch } from 'react-redux';

import { updateRecommendersOrder, duplicateRecommender } from '../../reducers/recommenders';
import { openConfirmDialog, openRecEditDialog } from '../../reducers/dialogs';

const ITEM_TYPE = 'recommender';

function RecEditView({ index, name }) {
  const dispatch = useDispatch();
  const dragRef = useRef(null);
  const previewRef = useRef(null);

  const handleMove = (dragIndex, hoverIndex) => {
    dispatch(updateRecommendersOrder({ dragIndex, hoverIndex }));
  };

  const [{ handlerId }, drop] = useDrop({
    accept: ITEM_TYPE,
    collect(monitor) {
      return {
        handlerId: monitor.getHandlerId(),
      };
    },
    hover(item, monitor) {
      if (!previewRef.current) {
        return;
      }

      const dragIndex = item.index;
      const hoverIndex = index;

      if (dragIndex === hoverIndex) {
        return;
      }

      const hoverBoundingRect = previewRef.current?.getBoundingClientRect();
      const hoverMiddleY = (hoverBoundingRect.bottom - hoverBoundingRect.top) / 2;
      const clientOffset = monitor.getClientOffset();
      const hoverClientY = clientOffset.y - hoverBoundingRect.top;

      if (dragIndex < hoverIndex && hoverClientY <= hoverMiddleY) {
        return;
      }

      if (dragIndex > hoverIndex && hoverClientY > hoverMiddleY) {
        return;
      }

      handleMove(dragIndex, hoverIndex);

      item.index = hoverIndex;
    },
  });

  const [{ isDragging }, drag, preview] = useDrag({
    type: ITEM_TYPE,
    item: () => ({ index }),
    collect: (monitor) => ({
      isDragging: monitor.isDragging(),
    }),
  });

  drag(dragRef);
  drop(preview(previewRef));

  const handleDelete = () => {
    dispatch(
      openConfirmDialog({
        title: 'Delete this recommender?',
        content: 'Deleting this recommender all settings will be lost.',
        params: {
          index,
        },
      })
    );
  };

  const handleEdit = () => {
    dispatch(openRecEditDialog(index));
  };

  const handleDuplicate = () => {
    dispatch(duplicateRecommender(index));
  };

  return (
    <Paper
      data-handler-id={handlerId}
      ref={previewRef}
      sx={{
        width: '100%',
        opacity: isDragging ? 0.4 : 1,
        display: 'flex',
        justifyContent: 'space-between',
        p: 2,
      }}
    >
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
        }}
      >
        <Box
          component="span"
          ref={dragRef}
          sx={{
            cursor: 'grab',
            marginRight: 1,
          }}
        >
          <DragIndicatorIcon
            sx={{
              display: 'block',
            }}
          />
        </Box>
        <Typography variant="h6" component="div">
          {name}
        </Typography>
      </Box>
      <Stack direction="row" spacing={1}>
        <IconButton onClick={handleEdit}>
          <EditIcon />
        </IconButton>
        <IconButton onClick={handleDuplicate}>
          <ContentCopyIcon />
        </IconButton>
        <IconButton onClick={handleDelete}>
          <DeleteIcon />
        </IconButton>
      </Stack>
    </Paper>
  );
}

RecEditView.propTypes = {
  name: pt.string.isRequired,
  index: pt.number.isRequired,
};

export default RecEditView;
