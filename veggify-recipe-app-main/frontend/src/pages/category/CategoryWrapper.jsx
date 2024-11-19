import React from 'react';

function CategoryItem({ name, href, backgroundColor, color }) {
  const style = {
    backgroundColor: backgroundColor,
    color: color,
    borderColor: color
  };

  return (
    <div >
      <a href={href} className=" rounded-full">
        <div className="category-tag uppercase px-6 py-2 text-center rounded-full" style={style}>{name}</div>
      </a>
    </div>
  );
}

function CategoryList() {
  return (
    <div className="flex flex-wrap items-center justify-center gap-8">
      <CategoryItem name="entrees" href="/categories/entrees" backgroundColor="#f0f5c4" color="#59871f" />
      <CategoryItem name="breakfast" href="/categories/breakfast" backgroundColor="#efedfa" color="#3c3a8f" />
      <CategoryItem name="lunch" href="/categories/lunch" backgroundColor="#e5f7f3" color="#1f8787" />
      <CategoryItem name="desserts" href="/categories/desserts" backgroundColor="#e8f5fa" color="#397a9e" />
      <CategoryItem name="sides" href="/categories/sides" backgroundColor="#feefc9" color="#d16400" />
      <CategoryItem name="drinks" href="/categories/drinks" backgroundColor="#ffeae3" color="#f0493e" />
    </div>
  );
}

function CategoryWrapper() {
  return (
    <div className="category-wrapper w-dyn-list">
      <CategoryList />
    </div>
  );
}

export default CategoryWrapper;
