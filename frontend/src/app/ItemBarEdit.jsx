/* eslint-disable no-param-reassign */
import React, { useRef } from 'react';
import pt from 'prop-types';
import Typography from '@mui/material/Typography';
import Paper from '@mui/material/Paper';
import Box from '@mui/material/Box';
import { useDrag, useDrop } from 'react-dnd';
import DragIndicatorIcon from '@mui/icons-material/DragIndicator';
import EditIcon from '@mui/icons-material/Edit';
import DeleteIcon from '@mui/icons-material/Delete';
import IconButton from '@mui/material/IconButton';

const DND_ITEM_TYPE = 'itemBar';

function ItemBarPreview({ index, title, onMove, onDelete }) {
  const dragRef = useRef(null);
  const previewRef = useRef(null);

  const [{ handlerId }, drop] = useDrop({
    accept: DND_ITEM_TYPE,
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

      onMove(dragIndex, hoverIndex);

      item.index = hoverIndex;
    },
  });

  const [{ isDragging }, drag, preview] = useDrag({
    type: DND_ITEM_TYPE,
    item: () => ({ index }),
    collect: (monitor) => ({
      isDragging: monitor.isDragging(),
    }),
  });

  drag(dragRef);
  drop(preview(previewRef));

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
        <Typography variant="h6">{title}</Typography>
      </Box>
      <Box>
        <IconButton>
          <EditIcon />
        </IconButton>
        <IconButton onClick={() => onDelete(index)}>
          <DeleteIcon />
        </IconButton>
      </Box>
    </Paper>
  );
}

ItemBarPreview.defaultProps = {};

ItemBarPreview.propTypes = {
  title: pt.string.isRequired,
  onMove: pt.func.isRequired,
  onDelete: pt.func.isRequired,
  index: pt.number.isRequired,
};

export default ItemBarPreview;
