import React, { useState, useMemo, useRef, useEffect } from 'react';
import { Paper, Stack, Backdrop, Box, CircularProgress, TextField } from '@mui/material';
import Plotly from 'plotly.js';

import { ScatterPlot } from '../plots';
import { CategoryFilter } from '../filters';
import { plotColors } from '../../const';

const userEmbeddings = [
  {
    id: 1,
    x: 0.89,
    y: 0.4,
  },
  {
    id: 2,
    x: 0.3,
    y: 0.29,
  },
  {
    id: 3,
    x: 0.41,
    y: 0.25,
  },
  {
    id: 4,
    x: 0.56,
    y: 0.59,
  },
  {
    id: 5,
    x: 0.37,
    y: 0.73,
  },
  {
    id: 6,
    x: 0.32,
    y: 0.33,
  },
  {
    id: 7,
    x: 0.68,
    y: 0.95,
  },
  {
    id: 8,
    x: 0.25,
    y: 0.44,
  },
  {
    id: 9,
    x: 0.71,
    y: 0.39,
  },
  {
    id: 10,
    x: 0.35,
    y: 0.87,
  },
  {
    id: 11,
    x: 0.19,
    y: 0.16,
  },
  {
    id: 12,
    x: 0.47,
    y: 0.26,
  },
  {
    id: 13,
    x: 0.46,
    y: 0.99,
  },
  {
    id: 14,
    x: 0.44,
    y: 0.32,
  },
  {
    id: 15,
    x: 0.4,
    y: 0.9,
  },
  {
    id: 16,
    x: 0.28,
    y: 0.78,
  },
  {
    id: 17,
    x: 0.54,
    y: 0.08,
  },
  {
    id: 18,
    x: 0.13,
    y: 0.73,
  },
  {
    id: 19,
    x: 0.42,
    y: 0.27,
  },
  {
    id: 20,
    x: 0.16,
    y: 0.11,
  },
  {
    id: 21,
    x: 0.8,
    y: 0.11,
  },
  {
    id: 22,
    x: 0.21,
    y: 0.89,
  },
  {
    id: 23,
    x: 0.15,
    y: 0.83,
  },
  {
    id: 24,
    x: 0.84,
    y: 0.19,
  },
  {
    id: 25,
    x: 0.16,
    y: 0.13,
  },
  {
    id: 26,
    x: 0.96,
    y: 0.76,
  },
  {
    id: 27,
    x: 0.32,
    y: 0.39,
  },
  {
    id: 28,
    x: 0.52,
    y: 0.99,
  },
  {
    id: 29,
    x: 0.18,
    y: 0.93,
  },
  {
    id: 30,
    x: 0.68,
    y: 0.34,
  },
];

const itemEmbeddings = [
  {
    x: 0.72,
    y: 0.7,
    c: 3,
    label: 'Ghosts of Mississippi',
    year: 1968,
    id: 1307,
    country: 'CO',
    genres: ['comedy', 'action'],
  },
  {
    x: 0.37,
    y: 0.78,
    c: 1,
    id: 1303,
    label: 'American Pie 2',
    year: 1996,
    country: 'CO',
    genres: ['action'],
  },
  {
    x: 0.07,
    y: 0.44,
    c: 5,
    id: 1302,
    label: 'Inglorious Bastards (Quel maledetto treno blindato)',
    year: 1997,
    country: 'MK',
    genres: ['comedy', 'horror', 'action'],
  },
  {
    x: 0.07,
    y: 0.93,
    c: 3,
    id: 1300,
    label: 'Prince of Egypt, The',
    year: 2001,
    country: 'FR',
    genres: ['action'],
  },
  {
    x: 0.63,
    y: 0.79,
    c: 3,
    id: 1322,
    label: 'Friday the 13th',
    year: 2010,
    country: 'CN',
    genres: ['horror'],
  },
  {
    x: 0.65,
    y: 0.4,
    c: 2,
    id: 1325,
    label: 'Pearl Harbor',
    year: 2015,
    country: 'ID',
    genres: ['drama'],
  },
  {
    x: 0.42,
    y: 0.79,
    c: 3,
    id: 1329,
    label: 'Ghosts of Mississippi',
    year: 1968,
    country: 'CO',
    genres: ['comedy', 'action'],
  },
  {
    x: 0.17,
    y: 0.98,
    c: 1,
    id: 13212,
    label: 'American Pie 2',
    year: 1996,
    country: 'CO',
    genres: ['action'],
  },
  {
    x: 0.37,
    y: 0.84,
    c: 5,
    id: 13232,
    label: 'Inglorious Bastards (Quel maledetto treno blindato)',
    year: 1997,
    country: 'MK',
    genres: ['comedy', 'horror', 'action'],
  },
  {
    x: 0.27,
    y: 0.43,
    c: 3,
    id: 13543,
    label: 'Prince of Egypt, The',
    year: 2001,
    country: 'FR',
    genres: ['action'],
  },
  {
    x: 0.43,
    y: 0.29,
    c: 3,
    id: 1312,
    label: 'Friday the 13th',
    year: 2010,
    country: 'CN',
    genres: ['horror'],
  },
  {
    x: 0.6,
    y: 0.44,
    c: 2,
    id: 13765,
    label: 'Pearl Harbor',
    year: 2015,
    country: 'ID',
    genres: ['drama'],
  },
];

function getItemIds(argumentName, argumentValue) {
  return [13765, 1312, 1325, 1322];
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function EmbeddingsPlot({ columns, onSelect, dataType }) {
  const [embeddingsData, setEmbeddingsData] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedField, setSelectedField] = useState('');
  const [selectedValues, setSelectedValues] = useState([]);
  const [minInteractions, setMinInteractions] = useState(5);
  const [highlightedPoints, setHighlightedPoints] = useState([]);
  const scatterRef = useRef();

  useEffect(() => {
    async function fetchData() {
      setIsLoading(true);
      await sleep(1000);
      setIsLoading(false);
    }
    if (dataType === 'users') {
      fetchData();
      setEmbeddingsData(userEmbeddings);
    } else {
      fetchData();
      setEmbeddingsData(itemEmbeddings);
    }
  }, []);

  const scatterColors = useMemo(() => {
    if (!highlightedPoints.length) {
      return plotColors.selectedMarker;
    }

    const colors = [];
    for (let i = 0; i < embeddingsData.length; i += 1) {
      colors.push(plotColors.unselectedMarker);
    }

    highlightedPoints.forEach((p) => {
      colors[p] = plotColors.selectedMarker;
    });

    return colors;
  }, [highlightedPoints]);

  const filterOptions = useMemo(() => {
    if (selectedField) {
      const col = columns[selectedField];
      if (col.dtype === 'number') {
        return col.bins.map((bin, index) => ({
          value: index,
          label: `${bin[0]} - ${bin[1]}`,
        }));
      }
      return col.options.map((option) => ({
        value: option,
        label: option,
      }));
    }
    return [];
  }, [selectedField]);

  const scatterPoints = useMemo(
    () => ({
      x: embeddingsData.map(({ x }) => x),
      y: embeddingsData.map(({ y }) => y),
      meta: embeddingsData.map(({ id }) => ({ id })),
      label: embeddingsData.map(({ label }) => label),
    }),
    [isLoading]
  );

  const isMultipleSelect = useMemo(() => {
    if (selectedField) {
      const fieldType = columns[selectedField].dtype;
      return fieldType === 'tags' || fieldType === 'category';
    }
    return false;
  }, [selectedField]);

  const resetFilterSelection = () => {
    setSelectedValues([]);
    setSelectedField('');
  };

  const resetScatterSelection = () => {
    Plotly.restyle(scatterRef.current.el, { selectedpoints: [null] });
  };

  const handleFilterApply = async () => {
    if (selectedValues.length) {
      setIsLoading(true);
      await sleep(500);
      setIsLoading(false);

      const ids = getItemIds(selectedField, selectedValues);

      if (!ids.length) {
        setHighlightedPoints([]);
      } else {
        const indices = embeddingsData.reduce((acc, item, index) => {
          if (ids.includes(item.id)) {
            acc.push(index);
          }
          return acc;
        }, []);
        setHighlightedPoints(indices);
      }
    }
  };

  const handleAttributeChange = (newValue) => {
    setSelectedField(newValue);
    setSelectedValues([]);
    setHighlightedPoints([]);
    resetScatterSelection();
    onSelect([]);
  };

  const handleValuesChange = (newValue) => {
    if (isMultipleSelect) {
      setSelectedValues(newValue);
    } else {
      setSelectedValues([newValue]);
    }
  };

  const handleScatterSelect = (eventData) => {
    if (eventData && eventData.points.length) {
      const { points } = eventData;
      const itemsIds = points.map((p) => p.customdata.id);
      onSelect(itemsIds);
      setHighlightedPoints(points[0].data.selectedpoints);
      resetFilterSelection();
    }
  };

  const handleScatterUnselect = () => {
    onSelect([]);
    setHighlightedPoints([]);
    resetScatterSelection();
    resetFilterSelection();
  };

  const handleMinInteractionsChange = (event) => {
    setMinInteractions(event.target.value);
  };

  return (
    <Paper sx={{ p: 2, height: '100%' }}>
      <Stack direction="row" spacing={2}>
        <CategoryFilter
          label="Item attribute"
          displayEmpty
          value={selectedField}
          onChange={handleAttributeChange}
          options={Object.keys(columns).map((col) => ({
            value: col,
            label: col,
          }))}
        />
        <CategoryFilter
          label="Attribute value"
          disabled={!selectedField}
          value={isMultipleSelect ? selectedValues : selectedValues[0]}
          multiple={isMultipleSelect}
          onClose={handleFilterApply}
          onChange={handleValuesChange}
          options={filterOptions}
        />
        {dataType === 'users' && (
          <TextField
            sx={{ minWidth: 250 }}
            disabled={!selectedField}
            label="Min. interactions"
            type="number"
            onBlur={handleFilterApply}
            inputProps={{ inputMode: 'numeric', pattern: '[0-9]*', min: 1 }}
            onChange={handleMinInteractionsChange}
            value={minInteractions}
            variant="filled"
          />
        )}
      </Stack>
      <Box position="relative">
        {isLoading && (
          <Backdrop
            sx={{
              position: 'absolute',
              color: '#000',
              backgroundColor: 'rgba(255,255,255,0.6)',
              zIndex: 100,
            }}
            open
          >
            <CircularProgress color="inherit" />
          </Backdrop>
        )}
        <ScatterPlot
          height={450}
          x={scatterPoints.x}
          y={scatterPoints.y}
          meta={scatterPoints.meta}
          color={scatterColors}
          label={scatterPoints.label}
          innerRef={scatterRef}
          onDeselect={handleScatterUnselect}
          onSelected={handleScatterSelect}
        />
      </Box>
    </Paper>
  );
}

export default EmbeddingsPlot;
