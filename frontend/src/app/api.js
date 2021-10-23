import { useState, useEffect } from 'react';

export function fetchItems(path, params = {}) {
  const [items, setItems] = useState([]);
  const [error, setError] = useState();
  const [isLoading, setIsLoading] = useState(true);

  const encodedParams = new URLSearchParams(params).toString();
  const fullPath = `/api${path}${encodedParams ? `?${encodedParams}` : ''}`;

  useEffect(() => {
    let isActive = true;

    setIsLoading(true);

    fetch(fullPath)
      .then((response) => response.json())
      .then((data) => {
        setTimeout(() => {
          if (isActive) {
            setItems(data);
            setIsLoading(false);
          }
        }, 300);
      })
      .catch((err) => {
        setIsLoading(false);
        setError(err);
      });

    return () => {
      isActive = false;
    };
  }, [fullPath]);

  return { items, isLoading, error };
}
